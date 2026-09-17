#!/usr/bin/env python3
"""Does folding the shared-expert half into the fused collective give the same
answer, and what does reading it cost?

WHY THIS FILE EXISTS. Folding only the ROUTED reduce into the epilogue LOST in
the engine: 5.53% at 8192 and 3.54% at 16384, because DeepSeek-V3 returns
shared_output + routed_output and the runner then all-reduced the shared half on
its own, at identical bytes to the single reduce the baseline did of the sum. So
the epilogue added a collective. The fix has to make the ONE multimem pass carry
out_scale*routed + shared, and this file is the correctness proof for that store
before any engine latency is read.

THE GATES, and why each is exact rather than a cosine. A cosine over 58 million
elements hides a dropped shard, and this kernel's shards are the thing most
likely to be wrong.

  G1 null residual is inert. The fold entry called with a null residual and
     out_scale 1.0 must be BITWISE what the routed-only fused arm produces.
     s[q] * 1.0f is exact in IEEE fp32, so anything else is a defect in the
     branch, not arithmetic.

  G2 residual only, exactly. Called with out_scale 0.0, the store reduces to
     bf16(s*0 + R) = R, so the kernel's own collective must reproduce a plain
     multimem all-reduce of R BITWISE. This is the gate that proves the residual
     is read at the right address for every row and column the comm blocks
     cover: a partition gap or an overlap changes bits here and nothing else in
     this file would see it.

  G3 combined, against exact arithmetic. fp64 reference built from the same
     per-rank bf16 partials: 2.5 * sum_r x_r + sum_r R_r. Reported next to the
     error of the path vLLM ships, which reduces in bf16, scales in bf16 and
     adds in bf16. The fold rounds once, after scaling in fp32 and adding the
     residual in fp32, so it should be at least as close to exact as shipping
     is. Worse would be a reason not to ship it; equal or better is the answer.

  G4 the residual read is the only new cost. Same binary, same chunks, same
     ncomm, one argument apart: fused with a null residual against fused with a
     real one. That difference is the extra 117 MB read at 8192, and it is what
     the removed collective has to pay for.

Contention is a correctness precondition for this kernel family and not hygiene:
under SM contention it returns a near orthogonal answer with rc=0, so the
foreign memory on the busiest device is printed with every run.

usage: torchrun --nproc_per_node=8 tp8_fold_gate.py
"""
import ctypes
import hashlib
import os
import statistics as st
import subprocess
import sys

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as sm

# The two binaries this gate compares, one argument apart in what they read.
# SO_FOLD folds the shared half in the kernel epilogue, which for the shipped
# DeepSeek geometry is kernel/build.sh deepseek_tp; SO_SHIP is the build before
# that fold. Explicit, because a gate that loaded the wrong side still passes.
SO_SHIP = os.environ["SO_SHIP"]
SO_FOLD = os.environ["SO_FOLD"]

LOCAL = int(os.environ.get("LOCAL_RANK", "0"))
DEV = f"cuda:{LOCAL}"

# DeepSeek-V3 at TP=8. N_HALF is moe_intermediate_size 2048 / 8, N_UP twice it.
E, K, N_UP, N_HALF, H_DIM, TOP_K = 256, 7168, 512, 256, 7168, 8
NK, NW, DK, DW = K // 128, N_UP // 128, N_HALF // 128, H_DIM // 128

# The engine's own table: 8192 tokens takes 4 chunks, 16384 takes 8. Anything
# else is a size the epilogue declines, so there is nothing to gate there.
SIZES = os.environ.get("SIZES", "8192:4,16384:8")
NCOMM = int(os.environ.get("NCOMM", "14"))
# DeepSeek-V3's routed_scaling_factor. Carried as the real value rather than a
# round number so the fp64 comparison is the one the engine will run.
SCALE = float(os.environ.get("SCALE", "2.5"))
ROUNDS = int(os.environ.get("ROUNDS", "25"))
WARM = int(os.environ.get("WARM", "8"))
PAD_WORDS = int(os.environ.get("PAD_WORDS", "65536"))

_seq = [0]


def probes(lib):
    g = {}
    for p in ("grid_size", "n_up", "n_half", "k_dim", "h_dim", "num_experts",
              "top_k", "router_mode", "max_tiles", "workspace_bytes", "block_m",
              "fused_renorm", "tp_max_c", "tp_align", "tp_mm", "tp_has_residual"):
        try:
            f = getattr(lib, "prefill_wgmma_" + p)
        except AttributeError:
            g[p] = None
            continue
        f.restype = ctypes.c_size_t if p == "workspace_bytes" else ctypes.c_int
        g[p] = f()
    return g


def foreign_mem(local):
    """Memory held on THIS device by processes other than mine."""
    try:
        o = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
             "--format=csv,noheader"], capture_output=True, text=True).stdout
        u = subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_uuid", "--format=csv,noheader",
             "-i", str(local)], capture_output=True, text=True).stdout.strip()
    except Exception:
        return -1
    mine, tot = os.getpid(), 0
    for line in o.strip().splitlines():
        if not line.strip():
            continue
        uuid, pid, mem = [x.strip() for x in line.split(",")]
        if uuid in u and int(pid) != mine:
            tot += int(mem.split()[0])
    return tot


def ulpstats(a, b):
    ai = a.view(torch.int16).to(torch.int32)
    bi = b.view(torch.int16).to(torch.int32)
    d = (ai - bi).abs()
    return dict(n=int((d > 0).sum()), maxulp=int(d.max()),
                nfar=int((d > 1).sum()))


def relerr(a, ref):
    """Max and mean relative error against an fp64 reference."""
    a = a.double().flatten()
    ref = ref.double().flatten()
    den = ref.abs().clamp(min=1e-6)
    r = (a - ref).abs() / den
    return float(r.max()), float(r.mean())


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(
        a.double().flatten(), b.double().flatten(), dim=0))


def agreed(ok):
    """True only if every rank says True. A per-rank PASS is not a PASS."""
    t = torch.tensor([1.0 if ok else 0.0], device=DEV)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return float(t) == 1.0


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(LOCAL)
    gname = dist.group.WORLD.group_name

    lib_s = ctypes.CDLL(SO_SHIP)
    lib_f = ctypes.CDLL(SO_FOLD)
    lib_s.launch_prefill_moe_wgmma_q1_tp.restype = ctypes.c_int
    lib_f.launch_prefill_moe_wgmma_q1_tp.restype = ctypes.c_int
    lib_f.launch_prefill_moe_wgmma_q1_tp2.restype = ctypes.c_int

    g_s, g_f = probes(lib_s), probes(lib_f)
    if not g_f["tp_has_residual"]:
        raise SystemExit(f"{SO_FOLD} does not export the residual entry point")
    md5 = {n: hashlib.md5(open(p, "rb").read()).hexdigest()
           for n, p in (("ship", SO_SHIP), ("fold", SO_FOLD))}
    if md5["ship"] == md5["fold"]:
        raise SystemExit("the two arms are the SAME binary")
    BM = g_f["block_m"]

    # Both binaries derive the sequence on device, and calling the graph-safe
    # entry with the old argument count links fine and runs silently unsafe, so
    # this is read from the .so rather than from the arm name.
    def devseq(lib):
        try:
            return bool(lib.prefill_wgmma_tp_has_devseq())
        except AttributeError:
            return False

    DS, DF = devseq(lib_s), devseq(lib_f)
    if not DF:
        raise SystemExit("the fold binary has no device sequence counter")

    if rank == 0:
        print("=" * 100)
        print("Folding the shared-expert half into the fused collective: is it "
              "the same answer, and what does it cost?")
        print("=" * 100)
        for n, g in (("ship", g_s), ("fold", g_f)):
            print(f"  {n:>4} grid={g['grid_size']} BM={g['block_m']} "
                  f"n_up={g['n_up']} n_half={g['n_half']} tp_max_c={g['tp_max_c']} "
                  f"tp_align={g['tp_align']} residual={g['tp_has_residual']} "
                  f"md5={md5[n][:12]}")
        for f in ("grid_size", "n_up", "n_half", "k_dim", "h_dim", "num_experts",
                  "top_k", "router_mode", "block_m", "fused_renorm", "tp_max_c",
                  "tp_align", "tp_mm"):
            if g_s[f] != g_f[f]:
                raise SystemExit(f"arms disagree on {f}: {g_s[f]} vs {g_f[f]}")
        print("  geometry agrees on every field: the residual store is the only "
              "difference between these two binaries")
        print(f"  sizes={SIZES} ncomm={NCOMM} scale={SCALE} rounds={ROUNDS} "
              f"devseq ship={DS} fold={DF}")
        print("# CSV_F,tokens,arm,nchunk,ncomm,total_us")
        sys.stdout.flush()

    fm = torch.tensor([float(foreign_mem(LOCAL))], device=DEV)
    dist.all_reduce(fm, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"# foreign memory on the busiest rank's device: {int(fm)} MiB "
              f"(contention is a correctness precondition here)")
        sys.stdout.flush()

    lib_f.prefill_wgmma_tp_pad_words.restype = ctypes.c_int
    lib_f.prefill_wgmma_tp_pad_words.argtypes = [ctypes.c_int] * 3
    sizes = []
    for tok in SIZES.split(","):
        a, b = tok.split(":")
        sizes.append((int(a), int(b)))
    need = max(lib_f.prefill_wgmma_tp_pad_words(c, NCOMM, world)
               for _, c in sizes)
    if need > PAD_WORDS:
        raise SystemExit(f"barrier window too small: need {need}w have {PAD_WORDS}w")

    flags = sm.empty(PAD_WORDS, device=DEV, dtype=torch.int32)
    hf = sm.rendezvous(flags, gname)
    flags.zero_()
    pads = hf.buffer_ptrs_dev
    # Persistent and deliberately NOT in the kernel workspace: the launcher
    # memsets the workspace on every launch, so a counter kept there would reset
    # on every replay and never advance.
    seqctr = torch.zeros(1, device=DEV, dtype=torch.int32)
    seqctr_p = ctypes.c_void_p(seqctr.data_ptr())
    torch.cuda.synchronize()
    dist.barrier()

    torch.manual_seed(1234 + rank)
    up_w = torch.randn(E, N_UP, K, device=DEV, dtype=torch.bfloat16) * 0.02
    uf = up_w.float().view(E, NW, 128, NK, 128)
    usc = (uf.abs().amax(dim=(2, 4)) / 448.0).clamp(min=1e-12).to(torch.float32)
    uq = (uf / usc[:, :, None, :, None]).clamp(-448, 448).view(
        E, N_UP, K).to(torch.float8_e4m3fn).contiguous()
    del uf, up_w
    torch.cuda.empty_cache()
    dn_w = torch.randn(E, H_DIM, N_HALF, device=DEV, dtype=torch.bfloat16) * 0.02
    df = dn_w.float().view(E, DW, 128, DK, 128)
    dsc = (df.abs().amax(dim=(2, 4)) / 448.0).clamp(min=1e-12).to(torch.float32)
    dq = (df / dsc[:, :, None, :, None]).clamp(-448, 448).view(
        E, H_DIM, N_HALF).to(torch.float8_e4m3fn).contiguous()
    del df, dn_w
    torch.cuda.empty_cache()
    usc, dsc = usc.contiguous(), dsc.contiguous()
    ws_b = max(g_s["workspace_bytes"], g_f["workspace_bytes"])

    for BS, NCHUNK in sizes:
        hs = torch.randn(BS, K, device=DEV, dtype=torch.bfloat16) * 0.5
        rl = torch.randn(BS, E, device=DEV, dtype=torch.float32)
        ws = torch.zeros(ws_b, device=DEV, dtype=torch.uint8)
        af8 = torch.zeros(BS, K, device=DEV, dtype=torch.uint8)
        asc = torch.zeros(BS, NK, device=DEV, dtype=torch.float32)
        ti = torch.zeros(BS, TOP_K, device=DEV, dtype=torch.int32)
        tw = torch.zeros(BS, TOP_K, device=DEV, dtype=torch.float32)
        si = torch.zeros(BS * TOP_K, device=DEV, dtype=torch.int32)
        oa = torch.zeros(BS * TOP_K, H_DIM, device=DEV, dtype=torch.float32)
        # The shared-expert half. Rank-local and PARTIAL, exactly like the routed
        # half, and the same order of magnitude as the routed output so the
        # comparison is not dominated by one of the two.
        resid = (torch.randn(BS, H_DIM, device=DEV, dtype=torch.bfloat16)
                 * 0.05).contiguous()
        bo = sm.empty(BS * H_DIM, device=DEV, dtype=torch.bfloat16)
        h = sm.rendezvous(bo, gname)
        mc = h.multicast_ptr
        if not mc:
            raise SystemExit("no multicast pointer, the fused path is unreachable")

        def base_of(lib):
            return [ctypes.c_void_p(rl.data_ptr()), ctypes.c_int(BS),
                    ctypes.c_int(BM),
                    ctypes.c_void_p(ws.data_ptr()), ctypes.c_size_t(ws_b),
                    ctypes.c_void_p(hs.data_ptr()),
                    ctypes.c_void_p(af8.data_ptr()),
                    ctypes.c_void_p(asc.data_ptr()),
                    ctypes.c_void_p(uq.view(torch.uint8).data_ptr()),
                    ctypes.c_void_p(usc.data_ptr()),
                    ctypes.c_void_p(dq.view(torch.uint8).data_ptr()),
                    ctypes.c_void_p(dsc.data_ptr()),
                    ctypes.c_void_p(oa.data_ptr()),
                    ctypes.c_void_p(ti.data_ptr()),
                    ctypes.c_void_p(tw.data_ptr()),
                    ctypes.c_void_p(si.data_ptr()),
                    ctypes.c_void_p(bo.data_ptr())]

        def tp_call(lib, nchunk, ncomm, stream, ds):
            """The routed-only fused entry, on either binary."""
            _seq[0] += 1
            return lib.launch_prefill_moe_wgmma_q1_tp(
                *base_of(lib), ctypes.c_void_p(0),
                ctypes.c_void_p(mc if ncomm else 0),
                ctypes.c_void_p(pads if ncomm else 0),
                ctypes.c_int(rank), ctypes.c_int(world),
                ctypes.c_int(nchunk), ctypes.c_int(ncomm),
                ctypes.c_uint(_seq[0]),
                *([seqctr_p] if ds else []),
                ctypes.c_void_p(stream.cuda_stream))

        def tp2_call(nchunk, ncomm, stream, res, scale):
            """The fold entry. The seqctr slot is always present here, unlike
            the older entry where it was last before the stream and therefore
            optional."""
            _seq[0] += 1
            return lib_f.launch_prefill_moe_wgmma_q1_tp2(
                *base_of(lib_f), ctypes.c_void_p(0),
                ctypes.c_void_p(mc if ncomm else 0),
                ctypes.c_void_p(pads if ncomm else 0),
                ctypes.c_int(rank), ctypes.c_int(world),
                ctypes.c_int(nchunk), ctypes.c_int(ncomm),
                ctypes.c_uint(_seq[0]), seqctr_p,
                ctypes.c_void_p(res.data_ptr() if res is not None else 0),
                ctypes.c_float(scale),
                ctypes.c_void_p(stream.cuda_stream))

        cur = torch.cuda.current_stream()

        # ---- the routed-only fused output, on the SHIPPED binary ----
        bo.zero_()
        rc_s = tp_call(lib_s, NCHUNK, NCOMM, cur, DS)
        torch.cuda.synchronize()
        ref_fused = bo.clone()
        cov = float((ref_fused.view(BS, H_DIM).double().abs().sum(1) > 0)
                    .double().mean())

        # ---- G1: a null residual and scale 1.0 is the same store ----
        bo.zero_()
        rc_n = tp2_call(NCHUNK, NCOMM, cur, None, 1.0)
        torch.cuda.synchronize()
        g1_null = bool(torch.equal(ref_fused, bo))
        # and the old entry on the new binary, which forwards nullptr and 1.0f
        bo.zero_()
        rc_o = tp_call(lib_f, NCHUNK, NCOMM, cur, DF)
        torch.cuda.synchronize()
        g1_fwd = bool(torch.equal(ref_fused, bo))
        ok1 = (g1_null and g1_fwd and rc_s == 0 and rc_n == 0 and rc_o == 0
               and cov >= 0.9999)
        a1 = agreed(ok1)
        if rank == 0:
            print(f"# G1 tok={BS} C={NCHUNK} rc={rc_s}/{rc_n}/{rc_o} cov={cov:.4f} "
                  f"null_residual_bitwise={g1_null} old_entry_bitwise={g1_fwd} "
                  f"-> {'PASS' if a1 else 'FAIL'}", flush=True)

        # ---- G2: out_scale 0.0 leaves the residual alone, exactly ----
        bo.copy_(resid.view(-1))
        torch.ops.symm_mem.multimem_all_reduce_(bo, "sum", gname)
        torch.cuda.synchronize()
        ref_resid = bo.clone()
        bo.zero_()
        rc_r = tp2_call(NCHUNK, NCOMM, cur, resid, 0.0)
        torch.cuda.synchronize()
        g2 = bool(torch.equal(ref_resid, bo))
        u2 = ulpstats(bo, ref_resid)
        a2 = agreed(g2 and rc_r == 0)
        if rank == 0:
            print(f"# G2 tok={BS} residual only, scale 0: bitwise={g2} "
                  f"moved={u2['n']} maxulp={u2['maxulp']} "
                  f"-> {'PASS' if a2 else 'FAIL'}", flush=True)

        # ---- G3: the combined store against exact arithmetic ----
        # The per-rank partials, reduced in fp64. dist.all_reduce over float64
        # is a different algorithm from either bf16 path, which is the point:
        # both are then measured against it.
        parts = ref_fused  # placeholder, replaced below
        del parts
        bo.zero_()
        rc_p = tp_call(lib_s, NCHUNK, 0, cur, DS)   # no reduce: the rank's own half
        torch.cuda.synchronize()
        local_routed = bo.view(BS, H_DIM).clone()
        r64 = local_routed.double()
        dist.all_reduce(r64)
        s64 = resid.double()
        dist.all_reduce(s64)
        exact = SCALE * r64 + s64

        # What vLLM ships: reduce in bf16, scale in bf16, add in bf16.
        bo.copy_(local_routed.view(-1))
        torch.ops.symm_mem.multimem_all_reduce_(bo, "sum", gname)
        torch.cuda.synchronize()
        red_routed = bo.view(BS, H_DIM).clone()
        shipped = ((red_routed.double() * SCALE).to(torch.bfloat16).double()
                   + ref_resid.view(BS, H_DIM).double()).to(torch.bfloat16)

        bo.zero_()
        rc_c = tp2_call(NCHUNK, NCOMM, cur, resid, SCALE)
        torch.cuda.synchronize()
        folded = bo.view(BS, H_DIM).clone()

        f_max, f_mean = relerr(folded, exact)
        s_max, s_mean = relerr(shipped, exact)
        # The fold rounds once where shipping rounds three times, so its error
        # must not be the larger of the two. A tolerance is still stated because
        # a single bf16 element can round the other way on a tie.
        ok3 = (rc_p == 0 and rc_c == 0 and f_mean <= s_mean * 1.05)
        a3 = agreed(ok3)
        if rank == 0:
            print(f"# G3 tok={BS} vs fp64 exact: fold max_rel={f_max:.3e} "
                  f"mean_rel={f_mean:.3e} cos={cos(folded, exact):.9f} | "
                  f"shipped max_rel={s_max:.3e} mean_rel={s_mean:.3e} "
                  f"cos={cos(shipped, exact):.9f} "
                  f"-> {'PASS' if a3 else 'FAIL'}", flush=True)
            print(f"#    fold vs shipped: {ulpstats(folded, shipped)} "
                  f"cos={cos(folded, shipped):.9f}", flush=True)

        if not (a1 and a2 and a3):
            if rank == 0:
                print(f"# tok={BS} GATE FAILED, no timing for this size",
                      flush=True)
            del hs, rl, ws, af8, asc, ti, tw, si, oa, bo, resid
            torch.cuda.empty_cache()
            continue

        # ---- G4: what the residual read costs ----
        def timeit(tag, run):
            for _ in range(WARM):
                run()
            torch.cuda.synchronize()
            dist.barrier()
            s = []
            for _ in range(ROUNDS):
                dist.barrier()
                torch.cuda.synchronize()
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record(cur)
                run()
                e1.record(cur)
                torch.cuda.synchronize()
                s.append(e0.elapsed_time(e1) * 1000.0)
            m = st.median(s)
            loc = torch.tensor([m], device=DEV)
            dist.all_reduce(loc, op=dist.ReduceOp.MAX)
            if rank == 0:
                print(f"CSV_F,{BS},{tag},{NCHUNK},{NCOMM},{float(loc):.1f}",
                      flush=True)
            return float(loc)

        t_ship = timeit("fused_ship", lambda: tp_call(lib_s, NCHUNK, NCOMM, cur, DS))
        t_null = timeit("fold_null", lambda: tp2_call(NCHUNK, NCOMM, cur, None, 1.0))
        t_fold = timeit("fold_resid",
                        lambda: tp2_call(NCHUNK, NCOMM, cur, resid, SCALE))
        # And the collective the engine would otherwise call on the shared half,
        # measured on the same buffer so the two are on one basis.
        t_extra = timeit("nccl_shared_only",
                         lambda: dist.all_reduce(bo))
        if rank == 0:
            mb = BS * H_DIM * 2 / 1e6
            print(f"# tok={BS} residual read costs "
                  f"{t_fold - t_null:+.1f} us on top of the same store with a "
                  f"null pointer ({mb:.1f} MB); the separate NCCL reduce of the "
                  f"shared half it removes is {t_extra:.1f} us; the new "
                  f"parameters cost {t_null - t_ship:+.1f} us on the shipped "
                  f"path", flush=True)

        del hs, rl, ws, af8, asc, ti, tw, si, oa, bo, resid
        del ref_fused, ref_resid, local_routed, r64, s64, exact, folded, shipped
        torch.cuda.empty_cache()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
