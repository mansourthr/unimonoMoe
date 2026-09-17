#!/usr/bin/env python3
"""DeepSeek-V3 prefill monokernel at TP=8: does the fused epilogue beat the
collective it replaces, on the whole kernel rather than on a Phase-3 replica?

WHAT THIS ADDS OVER THE PRIMITIVE TEST. An earlier standalone probe measured
Phase 3 plus the reduce in isolation and found the fused epilogue worth 127.8 us at
8192 tokens against its own serial control. That is a Phase-3-window number. It says
nothing about whether dedicating `ncomm` blocks changes phases 0 to 2, and phases
0 to 2 are 2409.5 of the 2663.6 us the kernel takes. This file runs the SHIPPING
kernel end to end at TP=8 and compares three arms over the same routing, the same
weights and the same symmetric output buffer:

  NCCL    shipping arm, then dist.all_reduce. The collective the engine calls.
  SYMM    shipping arm, then the in-place multimem reduce. Same bytes, same
          buffer, different collective. Isolates the collective from the fusion.
  FUSED   the tp8fuse arm through launch_prefill_moe_wgmma_q1_tp, which releases
          the reduce chunk by chunk from inside Phase 3.

WHAT THIS IS NOT. Every arm here writes into a symmetric window already, so the
copy-removal step is not in the comparison; NCCL is measured on a symmetric
buffer, not on the engine's staging copy. The engine number stays an engine
measurement.

THE GATE IS BITWISE AND IT RUNS FIRST.
  1. The tp8fuse arm with ncomm=0 must be bitwise equal to the shipping arm
     before any reduce. That is what proves the Phase-3 patch is inert on the
     path the shipping numbers were taken on.
  2. FUSED must be bitwise equal to SYMM. Both issue the same
     multimem.ld_reduce.add.acc::f32, whose per-instruction accumulation order is
     fixed by the hardware, so any shard partition must land on the same bits. A
     partition gap or overlap shows up here and nowhere in a cosine.
  3. NCCL is compared by ulp, not bitwise: it is a different algorithm, so a bit
     difference there is arithmetic and not a defect.
Contention is a correctness precondition for this kernel family, not hygiene, so
the per-device foreign memory is printed with every run.
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

from symm_output_pool import SymmOutputPool

# The two binaries this gate compares. SO_FUSE is the fused build under test,
# which for the shipped DeepSeek geometry is kernel/build.sh deepseek_tp; SO_SHIP
# is the unfused build it is measured against, an earlier compile of the same
# source with the epilogue disabled. Neither is guessed: a gate that loaded the
# wrong side would still print a pass.
SO_SHIP = os.environ["SO_SHIP"]
SO_FUSE = os.environ["SO_FUSE"]

LOCAL = int(os.environ.get("LOCAL_RANK", "0"))
DEV = f"cuda:{LOCAL}"

# DeepSeek-V3 at TP=8. N_HALF is moe_intermediate_size 2048 / 8, N_UP twice it.
E, K, N_UP, N_HALF, H_DIM, TOP_K = 256, 7168, 512, 256, 7168, 8
NK, NW, DK, DW = K // 128, N_UP // 128, N_HALF // 128, H_DIM // 128

# The same token counts the isolated primitive was measured at, so the end-to-end
# number can be put next to the Phase-3-window number without rescaling.
BS_LIST = [int(x) for x in os.environ.get("BSLIST", "512,2048,8192").split(",")]
CHUNKS = [int(x) for x in os.environ.get("CHUNKS", "4,8").split(",")]
NCOMMS = [int(x) for x in os.environ.get("NCOMMS", "12,14,16").split(",")]
ROUNDS = int(os.environ.get("ROUNDS", "25"))
WARM = int(os.environ.get("WARM", "8"))
PAD_WORDS = int(os.environ.get("PAD_WORDS", "65536"))
# POOL=1 times the persistent-pool path. Same arms, same weights, same
# routing, so the only difference from POOL=0 is where the output buffer
# comes from, which is what makes the two latencies comparable.
POOL = int(os.environ.get("POOL", "0"))
POOL_MAX = int(os.environ.get("POOL_MAX", "30720"))
NSLOTS = int(os.environ.get("NSLOTS", "1"))

_seq = [0]


def probes(lib):
    g = {}
    for p in ("grid_size", "n_up", "n_half", "k_dim", "h_dim", "num_experts",
              "top_k", "router_mode", "max_tiles", "workspace_bytes", "block_m",
              "fused_renorm", "tp_max_c", "tp_align", "tp_mm"):
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
    n = int((d > 0).sum())
    return dict(n=n, maxulp=int(d.max()), nfar=int((d > 1).sum()))


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0))


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(LOCAL)
    gname = dist.group.WORLD.group_name

    lib_s = ctypes.CDLL(SO_SHIP)
    lib_f = ctypes.CDLL(SO_FUSE)
    lib_s.launch_prefill_moe_wgmma_q1.restype = ctypes.c_int
    lib_f.launch_prefill_moe_wgmma_q1.restype = ctypes.c_int
    lib_f.launch_prefill_moe_wgmma_q1_tp.restype = ctypes.c_int

    # Does this binary derive the sequence on device? Read it from the .so, not
    # from the arm name: calling the graph-safe entry with the old argument count
    # links fine and runs silently graph-unsafe.
    try:
        DEVSEQ = bool(lib_f.prefill_wgmma_tp_has_devseq())
        SEQ_BYTES = int(lib_f.prefill_wgmma_tp_seq_bytes())
    except AttributeError:
        DEVSEQ, SEQ_BYTES = False, 0
    g_s, g_f = probes(lib_s), probes(lib_f)
    md5 = {n: hashlib.md5(open(p, "rb").read()).hexdigest()
           for n, p in (("ship", SO_SHIP), ("fuse", SO_FUSE))}
    if md5["ship"] == md5["fuse"]:
        raise SystemExit("the two arms are the SAME binary")
    GRID = g_f["grid_size"]
    BM = g_f["block_m"]

    if rank == 0:
        print("=" * 104)
        print("DeepSeek-V3 prefill monokernel at TP=8: fused epilogue vs the "
              "collective, whole kernel")
        print("=" * 104)
        for n, g in (("ship", g_s), ("fuse", g_f)):
            print(f"  {n:>5} grid={g['grid_size']} BM={g['block_m']} "
                  f"n_up={g['n_up']} n_half={g['n_half']} rmode={g['router_mode']} "
                  f"renorm={g['fused_renorm']} tp_max_c={g['tp_max_c']} "
                  f"tp_align={g['tp_align']} md5={md5[n][:12]}")
        for f in ("grid_size", "n_up", "n_half", "k_dim", "h_dim", "num_experts",
                  "top_k", "router_mode", "block_m", "fused_renorm"):
            if g_s[f] != g_f[f]:
                raise SystemExit(f"arms disagree on {f}: {g_s[f]} vs {g_f[f]}")
        print(f"  geometry agrees on every field except the fused epilogue")
        print(f"  rounds={ROUNDS} warm={WARM} chunks={CHUNKS} ncomms={NCOMMS}")
        print(f"  devseq={DEVSEQ} seq_bytes={SEQ_BYTES} pool={POOL} "
              f"pool_max={POOL_MAX if POOL else 0} slots={NSLOTS if POOL else 0}")
        print("# CSV_E,tokens,arm,nchunk,ncomm,total_us")
        sys.stdout.flush()

    fm = foreign_mem(LOCAL)
    fmt = torch.tensor([float(fm)], device=DEV)
    dist.all_reduce(fmt, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"# foreign memory on the busiest rank's device: {int(fmt)} MiB "
              f"(contention is a correctness precondition here)")
        sys.stdout.flush()

    # Barrier window, its own symmetric allocation. NOT a handle's signal pad:
    # MEASURED, writing a pad corrupts the state torch's own collective keeps
    # there and hangs it.
    # Ask the kernel how many words it will touch rather than re-deriving the
    # formula here; a driver that computes the size itself is a second source of
    # truth and the two drift.
    lib_f.prefill_wgmma_tp_pad_words.restype = ctypes.c_int
    lib_f.prefill_wgmma_tp_pad_words.argtypes = [ctypes.c_int] * 3
    need = max(lib_f.prefill_wgmma_tp_pad_words(c, n, world)
               for c in CHUNKS for n in NCOMMS)
    if need > PAD_WORDS:
        raise SystemExit(f"barrier window too small: need {need}w have {PAD_WORDS}w")
    pool, pool_init_s = None, 0.0
    if POOL:
        import time as _t
        # The pool sizes its pad window for the widest legal configuration, so it
        # must cover the widest one this sweep will actually ask for.
        pad_need = lib_f.prefill_wgmma_tp_pad_words(
            g_f["tp_max_c"], GRID - 1, world)
        if pad_need < need:
            raise SystemExit(f"pool pad window {pad_need}w below the sweep's "
                             f"widest {need}w")
        t_p0 = _t.time()
        pool = SymmOutputPool(gname, POOL_MAX, H_DIM, pad_need,
                              nslots=NSLOTS, device=DEV)
        torch.cuda.synchronize()
        dist.barrier()
        pool_init_s = _t.time() - t_p0
        pads = pool.pads_ptr
        seqctr_p = ctypes.c_void_p(pool.seqctr_ptr if DEVSEQ else 0)
        if rank == 0:
            print(f"# pool init {pool_init_s*1e3:.1f} ms {pool.describe()}")
            sys.stdout.flush()
    else:
        flags = sm.empty(PAD_WORDS, device=DEV, dtype=torch.int32)
        hf = sm.rendezvous(flags, gname)
        flags.zero_()

        # Persistent device sequence counter. Deliberately NOT symmetric and NOT
        # in the kernel workspace: the launcher memsets the workspace on every
        # launch and a CUDA graph replays that memset, so a counter kept there
        # would reset on every replay and never advance. Zeroed once here and
        # never touched again from the host.
        seqctr = torch.zeros(1, device=DEV, dtype=torch.int32)
        seqctr_p = ctypes.c_void_p(seqctr.data_ptr() if DEVSEQ else 0)
        pads = hf.buffer_ptrs_dev
    if rank == 0:
        print(f"# barrier window "
              f"{pool.pad_words if POOL else PAD_WORDS}w, widest config "
              f"touches {need}w")
        sys.stdout.flush()
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

    for BS in BS_LIST:
        hs = (torch.randn(BS, K, device=DEV, dtype=torch.bfloat16) * 0.5)
        rl = torch.randn(BS, E, device=DEV, dtype=torch.float32)
        ws = torch.zeros(ws_b, device=DEV, dtype=torch.uint8)
        af8 = torch.zeros(BS, K, device=DEV, dtype=torch.uint8)
        asc = torch.zeros(BS, NK, device=DEV, dtype=torch.float32)
        ti = torch.zeros(BS, TOP_K, device=DEV, dtype=torch.int32)
        tw = torch.zeros(BS, TOP_K, device=DEV, dtype=torch.float32)
        si = torch.zeros(BS * TOP_K, device=DEV, dtype=torch.int32)
        oa = torch.zeros(BS * TOP_K, H_DIM, device=DEV, dtype=torch.float32)
        if POOL:
            # Neither an allocation nor a collective per shape: the buffer was
            # rendezvoused once at init and this shape takes its first
            # BS * H_DIM elements at offset 0.
            bo = pool.output(BS)
            mc = pool.mc_ptr()
        else:
            bo = sm.empty(BS * H_DIM, device=DEV, dtype=torch.bfloat16)
            h = sm.rendezvous(bo, gname)
            mc = h.multicast_ptr
        if not mc:
            raise SystemExit("no multicast pointer, the fused path is unreachable")

        base = [ctypes.c_void_p(rl.data_ptr()), ctypes.c_int(BS), ctypes.c_int(BM),
                ctypes.c_void_p(ws.data_ptr()), ctypes.c_size_t(ws_b),
                ctypes.c_void_p(hs.data_ptr()),
                ctypes.c_void_p(af8.data_ptr()), ctypes.c_void_p(asc.data_ptr()),
                ctypes.c_void_p(uq.view(torch.uint8).data_ptr()),
                ctypes.c_void_p(usc.data_ptr()),
                ctypes.c_void_p(dq.view(torch.uint8).data_ptr()),
                ctypes.c_void_p(dsc.data_ptr()),
                ctypes.c_void_p(oa.data_ptr()), ctypes.c_void_p(ti.data_ptr()),
                ctypes.c_void_p(tw.data_ptr()), ctypes.c_void_p(si.data_ptr()),
                ctypes.c_void_p(bo.data_ptr())]

        def ship(stream):
            return lib_s.launch_prefill_moe_wgmma_q1(
                *base, ctypes.c_void_p(stream.cuda_stream))

        def fuse(nchunk, ncomm, stream):
            _seq[0] += 1
            return lib_f.launch_prefill_moe_wgmma_q1_tp(
                *base, ctypes.c_void_p(0),
                ctypes.c_void_p(mc if ncomm else 0),
                ctypes.c_void_p(pads if ncomm else 0),
                ctypes.c_int(rank), ctypes.c_int(world),
                ctypes.c_int(nchunk), ctypes.c_int(ncomm),
                ctypes.c_uint(_seq[0]),
                *([seqctr_p] if DEVSEQ else []),
                ctypes.c_void_p(stream.cuda_stream))

        cur = torch.cuda.current_stream()

        # ---- gate 1: the patch is inert on the shipped path ----
        bo.zero_()
        rc = ship(cur)
        torch.cuda.synchronize()
        ref_kernel = bo.clone()
        bo.zero_()
        rc0 = fuse(1, 0, cur)
        torch.cuda.synchronize()
        inert = bool(torch.equal(ref_kernel, bo))
        cov = float((ref_kernel.view(BS, H_DIM).float().abs().sum(1) > 0)
                    .float().mean())

        # ---- reference reduced output: shipping kernel + multimem reduce ----
        bo.zero_()
        ship(cur)
        torch.ops.symm_mem.multimem_all_reduce_(bo, "sum", gname)
        torch.cuda.synchronize()
        ref_symm = bo.clone()
        # and the NCCL one, for the ulp comparison only
        bo.zero_()
        ship(cur)
        dist.all_reduce(bo)
        torch.cuda.synchronize()
        ref_nccl = bo.clone()
        us = ulpstats(ref_nccl, ref_symm)

        ok1 = inert and rc == 0 and rc0 == 0 and cov >= 0.9999
        flag = torch.tensor([1.0 if ok1 else 0.0], device=DEV)
        dist.all_reduce(flag)
        if rank == 0:
            print(f"# tok={BS} rc={rc}/{rc0} patch_inert={inert} cov={cov:.4f} "
                  f"| nccl vs symm: {us['n']} bf16 moved, max {us['maxulp']} ulp, "
                  f"{us['nfar']} over 1  cos={cos(ref_nccl, ref_symm):.7f} "
                  f"-> {'PASS' if float(flag) == world else 'FAIL'}")
            sys.stdout.flush()
        if float(flag) != world:
            del hs, rl, ws, af8, asc, ti, tw, si, oa, bo
            torch.cuda.empty_cache()
            continue

        def timeit(tag, run, nchunk, ncomm):
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
                print(f"CSV_E,{BS},{tag},{nchunk},{ncomm},{float(loc):.1f}",
                      flush=True)

        timeit("nccl", lambda: (ship(cur), dist.all_reduce(bo)), 0, 0)
        timeit("symm", lambda: (ship(cur),
                                torch.ops.symm_mem.multimem_all_reduce_(
                                    bo, "sum", gname)), 0, 0)
        # The control that separates the patch from the fusion: the fused BINARY
        # with the fused path switched off, plus the same external collective. If
        # this is level with symm then the extra parameters, the branch and the
        # larger Workspace cost nothing on the shipped path, and every difference
        # the fused arm shows is the fusion. If it is slower, that gap is a tax
        # the fusion has to pay back before any of its win is real.
        timeit("tp0", lambda: (fuse(1, 0, cur),
                               torch.ops.symm_mem.multimem_all_reduce_(
                                   bo, "sum", gname)), 1, 0)

        for nchunk in CHUNKS:
            for ncomm in NCOMMS:
                bo.zero_()
                rcf = fuse(nchunk, ncomm, cur)
                torch.cuda.synchronize()
                bit = bool(torch.equal(ref_symm, bo))
                mad = float((bo.float() - ref_symm.float()).abs().max())
                ok = bit and rcf == 0
                flag = torch.tensor([1.0 if ok else 0.0], device=DEV)
                dist.all_reduce(flag)
                allok = float(flag) == world
                if rank == 0:
                    print(f"# gate tok={BS} C={nchunk} ncomm={ncomm} rc={rcf} "
                          f"bitwise={bit} max_abs_diff={mad:.6g} "
                          f"-> {'PASS' if allok else 'FAIL'}", flush=True)
                if not allok:
                    continue
                timeit("fused", lambda: fuse(nchunk, ncomm, cur), nchunk, ncomm)

        # bo is a view of a pool slot under POOL=1, so dropping the name
        # here does not release the symmetric allocation. It must not.
        del hs, rl, ws, af8, asc, ti, tw, si, oa, bo, ref_kernel, ref_symm, ref_nccl
        torch.cuda.empty_cache()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
