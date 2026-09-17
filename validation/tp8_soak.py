#!/usr/bin/env python3
"""Adversarial validation of the fused epilogue: can it be made to hang, race, or
return a wrong answer?

The A/B driver proves the fused kernel is bitwise correct on a quiet stream with a
host synchronise around every launch. A serving engine does none of that. This
file attacks the four things that synchronise-every-launch hides:

  T1 SOAK          randomised (tokens, C, ncomm) for N iterations, bitwise checked
                   every time. Catches configuration-dependent partition bugs that
                   a fixed sweep misses, and anything that only shows up on the
                   hundredth launch.
  T2 BACK TO BACK  K fused launches enqueued with NO host sync between them, then
                   one check. Phase 3 rewrites bf16_output from output_accum before
                   reducing, so the operation is idempotent and the last launch's
                   result must still be exact. This is the only test that can catch
                   launch N+1 overwriting the output while a peer rank is still
                   reducing launch N.
  T3 SKEW          one rank is delayed by a spin kernel before the fused launch, so
                   its compute blocks arrive at the cross-rank barrier long after
                   everyone else's. Rotating which rank is late covers both sides of
                   every barrier. This is the shape a real engine produces whenever
                   ranks drift.
  T4 GRAPH         capture the fused launch into a CUDA graph and replay it. tp_seq
                   is a kernel ARGUMENT, so a replay reuses the captured value; the
                   barrier spins on `v >= seq`, which a second replay satisfies on
                   arrival. If that is what happens the reduce is released before the
                   peers have written and the result is WRONG, not hung. Checked, not
                   assumed.

A hang is as much a failure as a wrong answer and cannot be caught by an assert, so
every rank prints a heartbeat with an iteration number and the caller runs this
under a timeout.
"""
import ctypes
import os
import sys
import time

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
E, K, N_UP, N_HALF, H_DIM, TOP_K = 256, 7168, 512, 256, 7168, 8
NK, NW, DK, DW = K // 128, N_UP // 128, N_HALF // 128, H_DIM // 128
TOKS = [int(x) for x in os.environ.get("TOKS", "512,2048,8192,16384").split(",")]
CS = [int(x) for x in os.environ.get("CS", "1,2,3,4,5,8,16").split(",")]
NCS = [int(x) for x in os.environ.get("NCS", "1,4,8,12,14,16,20,32,64,131").split(",")]
ITERS = int(os.environ.get("ITERS", "300"))
BTB = int(os.environ.get("BTB", "8"))
GREPLAYS = int(os.environ.get("GREPLAYS", "8"))
TESTS = os.environ.get("TESTS", "T1,T2,T3,T4").split(",")
SEED = int(os.environ.get("SEED", "0"))
PAD_WORDS = 1 << 20
# POOL=1 sources the output buffer, the barrier pads and the sequence counter
# from the persistent pool instead of the harness's per-token-size hand
# rendezvous. Every test below is identical on both paths, so a difference in a
# result is a difference in the buffer scheme. POOL_MAX defaults to the engine's
# MOE_MONOKERNEL_PREFILL_MAX_TOKENS, and NSLOTS to the 2 that DBO would need.
POOL = int(os.environ.get("POOL", "0"))
POOL_MAX = int(os.environ.get("POOL_MAX", "30720"))
NSLOTS = int(os.environ.get("NSLOTS", "2"))

_seq = [0]


def lcg(state):
    """Deterministic and identical on every rank without touching Math.random or
    torch's generator, which the weight setup also uses."""
    while True:
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        yield state


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(LOCAL)
    gname = dist.group.WORLD.group_name
    lib_s, lib_f = ctypes.CDLL(SO_SHIP), ctypes.CDLL(SO_FUSE)
    lib_s.launch_prefill_moe_wgmma_q1.restype = ctypes.c_int
    lib_f.launch_prefill_moe_wgmma_q1_tp.restype = ctypes.c_int

    # Does this binary derive the sequence on device? Read it from the .so, not
    # from the arm name: calling the graph-safe entry with the old argument count
    # links fine and runs silently graph-unsafe.
    try:
        DEVSEQ = bool(lib_f.prefill_wgmma_tp_has_devseq())
        SEQ_BYTES = int(lib_f.prefill_wgmma_tp_seq_bytes())
    except AttributeError:
        DEVSEQ, SEQ_BYTES = False, 0
    for p in ("workspace_bytes",):
        getattr(lib_f, "prefill_wgmma_" + p).restype = ctypes.c_size_t
    ws_b = lib_f.prefill_wgmma_workspace_bytes()
    BM = lib_f.prefill_wgmma_block_m()
    GRID = lib_f.prefill_wgmma_grid_size()
    MAXC = lib_f.prefill_wgmma_tp_max_c()

    pool, pool_init_s = None, 0.0
    if POOL:
        # Everything with a lifetime longer than one launch is created here, once,
        # before any capture or timing: the output buffers, the barrier pads and
        # the sequence counter. Nothing after this line allocates, rendezvouses or
        # runs a collective on the harness's behalf, which is the property the
        # engine needs and the old per-shape rendezvous cannot have. The pad
        # window is sized by the BINARY's own formula at the widest legal
        # configuration, not by one re-derived here.
        lib_f.prefill_wgmma_tp_pad_words.restype = ctypes.c_int
        lib_f.prefill_wgmma_tp_pad_words.argtypes = [ctypes.c_int] * 3
        pad_need = lib_f.prefill_wgmma_tp_pad_words(MAXC, GRID - 1, world)
        t_p0 = time.time()
        pool = SymmOutputPool(gname, POOL_MAX, H_DIM, pad_need,
                              nslots=NSLOTS, device=DEV)
        torch.cuda.synchronize()
        dist.barrier()
        pool_init_s = time.time() - t_p0
        pads = pool.pads_ptr
        seqctr_p = ctypes.c_void_p(pool.seqctr_ptr if DEVSEQ else 0)
        if rank == 0:
            print(f"# pool init {pool_init_s*1e3:.1f} ms {pool.describe()}",
                  flush=True)
    else:
        flags = sm.empty(PAD_WORDS, device=DEV, dtype=torch.int32)
        hf = sm.rendezvous(flags, gname)
        flags.zero_()
        pads = hf.buffer_ptrs_dev

        # Persistent device sequence counter. Deliberately NOT symmetric and NOT
        # in the kernel workspace: the launcher memsets the workspace on every
        # launch and a CUDA graph replays that memset, so a counter kept there
        # would reset on every replay and never advance. Zeroed once here and
        # never touched again from the host.
        seqctr = torch.zeros(1, device=DEV, dtype=torch.int32)
        seqctr_p = ctypes.c_void_p(seqctr.data_ptr() if DEVSEQ else 0)
    torch.cuda.synchronize()
    dist.barrier()

    if rank == 0:
        print(f"# soak grid={GRID} block_m={BM} tp_max_c={MAXC} world={world} "
              f"iters={ITERS} btb={BTB} tests={TESTS} seed={SEED} "
              f"devseq={DEVSEQ} greplays={GREPLAYS}", flush=True)

    torch.manual_seed(1234 + rank)
    up_w = torch.randn(E, N_UP, K, device=DEV, dtype=torch.bfloat16) * 0.02
    uf = up_w.float().view(E, NW, 128, NK, 128)
    usc = (uf.abs().amax(dim=(2, 4)) / 448.0).clamp(min=1e-12).float()
    uq = (uf / usc[:, :, None, :, None]).clamp(-448, 448).view(
        E, N_UP, K).to(torch.float8_e4m3fn).contiguous()
    del uf, up_w
    torch.cuda.empty_cache()
    dn_w = torch.randn(E, H_DIM, N_HALF, device=DEV, dtype=torch.bfloat16) * 0.02
    df = dn_w.float().view(E, DW, 128, DK, 128)
    dsc = (df.abs().amax(dim=(2, 4)) / 448.0).clamp(min=1e-12).float()
    dq = (df / dsc[:, :, None, :, None]).clamp(-448, 448).view(
        E, H_DIM, N_HALF).to(torch.float8_e4m3fn).contiguous()
    del df, dn_w
    torch.cuda.empty_cache()
    usc, dsc = usc.contiguous(), dsc.contiguous()

    # One allocation set per token size, reused, so the soak does not spend its
    # time in the allocator and a pointer never changes under the kernel.
    T = {}
    for t in sorted(TOKS):
        T[t] = dict(
            hs=torch.randn(t, K, device=DEV, dtype=torch.bfloat16) * 0.5,
            rl=torch.randn(t, E, device=DEV, dtype=torch.float32),
            ws=torch.zeros(ws_b, device=DEV, dtype=torch.uint8),
            af8=torch.zeros(t, K, device=DEV, dtype=torch.uint8),
            asc=torch.zeros(t, NK, device=DEV, dtype=torch.float32),
            ti=torch.zeros(t, TOP_K, device=DEV, dtype=torch.int32),
            tw=torch.zeros(t, TOP_K, device=DEV, dtype=torch.float32),
            si=torch.zeros(t * TOP_K, device=DEV, dtype=torch.int32),
            oa=torch.zeros(t * TOP_K, H_DIM, device=DEV, dtype=torch.float32))
        if POOL:
            # No allocation and no collective per shape. Every slot is already
            # mapped, and a launch of t tokens takes the first t * H_DIM elements,
            # so every shape starts at the same 512-byte-aligned base (MEASURED)
            # and the shard alignment the reduce depends on needs no policy.
            T[t]["bo_s"] = [pool.output(t, s) for s in range(NSLOTS)]
            T[t]["mc_s"] = [pool.mc_ptr(s) for s in range(NSLOTS)]
        else:
            b = sm.empty(t * H_DIM, device=DEV, dtype=torch.bfloat16)
            h = sm.rendezvous(b, gname)
            T[t]["bo_s"] = [b]
            T[t]["mc_s"] = [h.multicast_ptr]
        assert all(T[t]["mc_s"]), "no multicast pointer"

    def bo(t, slot=0):
        """The output tensor a launch writes: one buffer per token size on the old
        path, one persistent pool slot on the new one. The tests cannot tell which
        they are running, which is the point."""
        return T[t]["bo_s"][slot]

    def mc(t, slot=0):
        return T[t]["mc_s"][slot]

    def base(t, slot=0):
        d = T[t]
        return [ctypes.c_void_p(d["rl"].data_ptr()), ctypes.c_int(t),
                ctypes.c_int(BM), ctypes.c_void_p(d["ws"].data_ptr()),
                ctypes.c_size_t(ws_b), ctypes.c_void_p(d["hs"].data_ptr()),
                ctypes.c_void_p(d["af8"].data_ptr()),
                ctypes.c_void_p(d["asc"].data_ptr()),
                ctypes.c_void_p(uq.view(torch.uint8).data_ptr()),
                ctypes.c_void_p(usc.data_ptr()),
                ctypes.c_void_p(dq.view(torch.uint8).data_ptr()),
                ctypes.c_void_p(dsc.data_ptr()),
                ctypes.c_void_p(d["oa"].data_ptr()),
                ctypes.c_void_p(d["ti"].data_ptr()),
                ctypes.c_void_p(d["tw"].data_ptr()),
                ctypes.c_void_p(d["si"].data_ptr()),
                ctypes.c_void_p(bo(t, slot).data_ptr())]

    def ship(t, stream):
        return lib_s.launch_prefill_moe_wgmma_q1(
            *base(t), ctypes.c_void_p(stream.cuda_stream))

    def fuse(t, C, nc, stream, seq=None, slot=0):
        if seq is None:
            _seq[0] += 1
            seq = _seq[0]
        return lib_f.launch_prefill_moe_wgmma_q1_tp(
            *base(t, slot), ctypes.c_void_p(0), ctypes.c_void_p(mc(t, slot)),
            ctypes.c_void_p(pads), ctypes.c_int(rank), ctypes.c_int(world),
            ctypes.c_int(C), ctypes.c_int(nc), ctypes.c_uint(seq),
            *([seqctr_p] if DEVSEQ else []),
            ctypes.c_void_p(stream.cuda_stream))

    cur = torch.cuda.current_stream()
    refs = {}
    for t in sorted(TOKS):
        # MEASURED: multimem_all_reduce_ accepts a prefix of a larger symmetric
        # allocation, so the reference runs on the same pool view the fused path
        # writes and no second buffer is needed to produce it.
        bo(t).zero_()
        assert ship(t, cur) == 0
        torch.ops.symm_mem.multimem_all_reduce_(bo(t), "sum", gname)
        torch.cuda.synchronize()
        refs[t] = bo(t).clone()
    dist.barrier()

    fails, checks, skipped = [], 0, 0
    worst = [0.0]

    def check(tag, t, C, nc, extra="", slot=0):
        nonlocal checks
        checks += 1
        o = bo(t, slot)
        ok = bool(torch.equal(refs[t], o))
        f = torch.tensor([0.0 if ok else 1.0], device=DEV)
        dist.all_reduce(f)
        if float(f) > 0:
            bad = int((refs[t] != o).sum())
            # Only computed on a failure. A passing check is torch.equal, so its
            # max_abs_diff is exactly 0 and reading it back would cost a second
            # full-tensor pass on every one of tens of thousands of checks.
            mad = float((o.float() - refs[t].float()).abs().max())
            worst[0] = max(worst[0], mad)
            fails.append((tag, t, C, nc, extra, int(float(f)), bad, mad))
            if rank == 0:
                print(f"FAIL {tag} tok={t} C={C} ncomm={nc} {extra} "
                      f"ranks_bad={int(float(f))} cells_bad_r0={bad} "
                      f"max_abs_diff={mad:.6g}", flush=True)
        return float(f) == 0

    # ---------------- T1 soak, randomised configuration ----------------
    if "T1" in TESTS:
        g = lcg(SEED * 7919 + 17)
        t0 = time.time()
        for i in range(ITERS):
            t = TOKS[next(g) % len(TOKS)]
            C = CS[next(g) % len(CS)]
            nc = NCS[next(g) % len(NCS)]
            if C > MAXC or nc >= GRID or (t * H_DIM) % (8 * C):
                skipped += 1
                continue
            bo(t).zero_()
            rc = fuse(t, C, nc, cur)
            torch.cuda.synchronize()
            if rc != 0:
                if rank == 0:
                    print(f"FAIL T1 rc={rc} tok={t} C={C} ncomm={nc}", flush=True)
                fails.append(("T1rc", t, C, nc, f"rc={rc}", 1, 0, 0.0))
                continue
            check("T1", t, C, nc)
            if rank == 0 and (i + 1) % 25 == 0:
                print(f"# T1 heartbeat {i+1}/{ITERS} checks={checks} "
                      f"fails={len(fails)} {time.time()-t0:.0f}s", flush=True)
        dist.barrier()
        if rank == 0:
            print(f"# T1 done checks={checks} fails={len(fails)} "
                  f"skipped={skipped}", flush=True)

    # ---------------- T2 back to back, no host sync ----------------
    if "T2" in TESTS:
        for t in sorted(TOKS):
            for C, nc in ((4, 14), (8, 16), (2, 12)):
                if (t * H_DIM) % (8 * C):
                    continue
                bo(t).zero_()
                for _ in range(BTB):
                    fuse(t, C, nc, cur)      # deliberately no synchronize
                torch.cuda.synchronize()
                check("T2", t, C, nc, f"btb={BTB}")
            if rank == 0:
                print(f"# T2 tok={t} done fails={len(fails)}", flush=True)
        dist.barrier()

    # ---------------- T3 rank skew ----------------
    if "T3" in TESTS:
        big = torch.randn(4096, 4096, device=DEV, dtype=torch.float32)
        for t in sorted(TOKS):
            for C, nc in ((4, 14), (8, 14)):
                if (t * H_DIM) % (8 * C):
                    continue
                for late in range(world):
                    bo(t).zero_()
                    if rank == late:
                        # ~1 ms of on-stream work ahead of the fused launch, so
                        # this rank reaches every barrier last.
                        for _ in range(24):
                            big = big @ big.t() * 1e-6
                    fuse(t, C, nc, cur)
                    torch.cuda.synchronize()
                    check("T3", t, C, nc, f"late_rank={late}")
            if rank == 0:
                print(f"# T3 tok={t} done fails={len(fails)}", flush=True)
        dist.barrier()

    # ---------------- T4 CUDA graph capture and replay ----------------
    # Three shapes, because a graph can fail in three different ways and the
    # first version of this test only covered the first:
    #   T4      plain repeated replay, the shape that exposed the baked-argument
    #           corruption.
    #   T4mix   replays and eager launches interleaved, which is what an engine
    #           produces when one batch shape is captured and another is not. A
    #           single counter has to keep both monotone.
    #   T4btb   replays enqueued back to back with no host sync, so replay N+1
    #           can be rewriting the output while a peer still reduces N.
    if "T4" in TESTS:
        for t in sorted(TOKS):
            for C, nc in ((4, 14), (8, 16)):
                if (t * H_DIM) % (8 * C):
                    continue
                gr = torch.cuda.CUDAGraph()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        fuse(t, C, nc, s)
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                dist.barrier()
                # With a device counter there is nothing to bake, so capture the
                # ordinary call. Without one, bake the next host value, which is
                # exactly the graph-unsafe shape the counter removes.
                cap = None if DEVSEQ else _seq[0] + 1
                if cap is not None:
                    _seq[0] = cap
                captured = True
                try:
                    with torch.cuda.graph(gr):
                        fuse(t, C, nc, torch.cuda.current_stream(), seq=cap)
                except Exception as e:
                    captured = False
                    if rank == 0:
                        print(f"# T4 capture REFUSED tok={t} C={C} ncomm={nc}: "
                              f"{type(e).__name__}: {e}", flush=True)
                if not captured:
                    continue
                for rep in range(GREPLAYS):
                    bo(t).zero_()
                    gr.replay()
                    torch.cuda.synchronize()
                    check("T4", t, C, nc,
                          f"replay={rep}" + ("" if DEVSEQ
                                             else f" seq_baked={cap}"))
                for rep in range(4):
                    bo(t).zero_()
                    gr.replay()
                    torch.cuda.synchronize()
                    check("T4mix", t, C, nc, f"graph={rep}")
                    bo(t).zero_()
                    fuse(t, C, nc, cur)
                    torch.cuda.synchronize()
                    check("T4mix", t, C, nc, f"eager={rep}")
                bo(t).zero_()
                for _ in range(BTB):
                    gr.replay()          # deliberately no synchronize
                torch.cuda.synchronize()
                check("T4btb", t, C, nc, f"btb={BTB}")
                del gr
            if rank == 0:
                print(f"# T4 tok={t} done fails={len(fails)}", flush=True)
        dist.barrier()

    # ---------------- T5 graph replay with a late rank ----------------
    if "T5" in TESTS:
        big5 = torch.randn(4096, 4096, device=DEV, dtype=torch.float32)
        for t in sorted(TOKS):
            for C, nc in ((4, 14), (8, 16)):
                if (t * H_DIM) % (8 * C):
                    continue
                gr = torch.cuda.CUDAGraph()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        fuse(t, C, nc, s)
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                dist.barrier()
                try:
                    with torch.cuda.graph(gr):
                        fuse(t, C, nc, torch.cuda.current_stream())
                except Exception as e:
                    if rank == 0:
                        print(f"# T5 capture REFUSED tok={t}: "
                              f"{type(e).__name__}: {e}", flush=True)
                    continue
                for late in range(world):
                    bo(t).zero_()
                    if rank == late:
                        for _ in range(24):
                            big5 = big5 @ big5.t() * 1e-6
                    gr.replay()
                    torch.cuda.synchronize()
                    check("T5", t, C, nc, f"late_rank={late}")
                del gr
            if rank == 0:
                print(f"# T5 tok={t} done fails={len(fails)}", flush=True)
        dist.barrier()

    # ---------------- T6 half the ranks replay, half launch eagerly ----------
    if "T6" in TESTS:
        for t in sorted(TOKS):
            for C, nc in ((4, 14), (8, 16)):
                if (t * H_DIM) % (8 * C):
                    continue
                gr = torch.cuda.CUDAGraph()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        fuse(t, C, nc, s)
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                dist.barrier()
                try:
                    with torch.cuda.graph(gr):
                        fuse(t, C, nc, torch.cuda.current_stream())
                except Exception as e:
                    if rank == 0:
                        print(f"# T6 capture REFUSED tok={t}: "
                              f"{type(e).__name__}: {e}", flush=True)
                    continue
                # Three splits: even ranks replay, odd ranks replay, then only
                # rank 0 replays. Each rank's local sequence therefore diverges
                # from its peers' in value while the counts stay equal.
                for split, pred in enumerate((lambda r: r % 2 == 0,
                                              lambda r: r % 2 == 1,
                                              lambda r: r == 0)):
                    for rep in range(3):
                        bo(t).zero_()
                        if pred(rank):
                            gr.replay()
                        else:
                            fuse(t, C, nc, cur)
                        torch.cuda.synchronize()
                        check("T6", t, C, nc, f"split={split} rep={rep}")
                del gr
            if rank == 0:
                print(f"# T6 tok={t} done fails={len(fails)}", flush=True)
        dist.barrier()

    # ---------------- T7 alternating sizes through one persistent buffer ------
    # The old path gave every token size its own symmetric allocation, so a small
    # launch could never land on top of a large one. With one buffer per slot that
    # ordering is producible and has to be tested: descending, ascending, then
    # big/small/big through the same allocation.
    #
    # The second half asserts the range discipline the pool depends on. The whole
    # slot is seeded with a sentinel and the bytes past num_tokens * H_DIM must
    # come back untouched, because in an engine those bytes are another request's
    # output. A kernel that derived any bound from the allocation rather than from
    # num_tokens would fail here and nowhere else.
    if "T7" in TESTS:
        order = (sorted(TOKS, reverse=True) + sorted(TOKS)
                 + [max(TOKS), min(TOKS), max(TOKS), min(TOKS), max(TOKS)])
        for C, nc in ((4, 14), (8, 16)):
            if C > MAXC or nc >= GRID:
                continue
            for t in order:
                if (t * H_DIM) % (8 * C):
                    continue
                bo(t).zero_()
                fuse(t, C, nc, cur)
                torch.cuda.synchronize()
                check("T7alt", t, C, nc, "alternating")
        if POOL:
            full = pool.slot_buffer(0)
            SENT = -7.0
            for t in sorted(TOKS):
                C, nc = 4, 14
                if (t * H_DIM) % (8 * C):
                    continue
                full.fill_(SENT)
                bo(t).zero_()
                fuse(t, C, nc, cur)
                torch.cuda.synchronize()
                check("T7tail", t, C, nc, "in_range")
                tail = full[t * H_DIM:]
                stray = int((tail != SENT).sum()) if tail.numel() else 0
                gt = torch.tensor([float(stray)], device=DEV)
                dist.all_reduce(gt, op=dist.ReduceOp.MAX)
                checks += 1
                if float(gt) > 0:
                    fails.append(("T7range", t, C, nc, "wrote past num_tokens",
                                  1, int(float(gt)), 0.0))
                    if rank == 0:
                        print(f"FAIL T7range tok={t} tail_cells_written="
                              f"{int(float(gt))} of {tail.numel()}", flush=True)
                elif rank == 0:
                    print(f"# T7range tok={t} tail {tail.numel()} cells "
                          f"untouched", flush=True)
        if rank == 0:
            print(f"# T7 done checks={checks} fails={len(fails)}", flush=True)
        dist.barrier()

    # ---------------- T8 pointer stability and a non-allocating hot path ------
    # A graph bakes the output pointer, so anything that moves the pool's
    # allocations after capture is a use-after-free the kernel cannot detect.
    # And a per-call allocation is precisely what makes the engine's current op
    # unusable here, so the absence of one is a property to measure and not to
    # assume: 64 launches must move allocated bytes, reserved bytes and the live
    # allocation COUNT by zero.
    if "T8" in TESTS and POOL:
        pool.assert_stable()
        t, C, nc = max(TOKS), 8, 16
        for _ in range(3):
            fuse(t, C, nc, cur)
        torch.cuda.synchronize()
        keys = ("allocated", "reserved", "live_allocations")
        before = (torch.cuda.memory_allocated(LOCAL),
                  torch.cuda.memory_reserved(LOCAL),
                  torch.cuda.memory_stats()["allocation.all.current"])
        for _ in range(64):
            fuse(t, C, nc, cur)
        torch.cuda.synchronize()
        after = (torch.cuda.memory_allocated(LOCAL),
                 torch.cuda.memory_reserved(LOCAL),
                 torch.cuda.memory_stats()["allocation.all.current"])
        pool.assert_stable()
        d = torch.tensor([float(abs(a - b)) for a, b in zip(after, before)],
                         device=DEV)
        dist.all_reduce(d, op=dist.ReduceOp.MAX)
        checks += 1
        if float(d.max()) > 0:
            fails.append(("T8alloc", t, C, nc,
                          "hot path allocated: "
                          + " ".join(f"{k}+{float(x):.0f}"
                                     for k, x in zip(keys, d)), 1, 0, 0.0))
            if rank == 0:
                print("FAIL T8alloc " + " ".join(f"{k}_delta={float(x):.0f}"
                                                 for k, x in zip(keys, d)),
                      flush=True)
        elif rank == 0:
            print("# T8 64 launches: allocated, reserved and live-allocation "
                  "count all delta 0 on every rank, pool pointers stable",
                  flush=True)
        check("T8", t, C, nc, "after_64_reuse")
        dist.barrier()

    # ---------------- T9 the slot policy is a requirement, not a convention ---
    # Two claims, and the second is what makes the first a requirement:
    #   T9uni  every rank on the same slot is bitwise correct, on each slot and
    #          when the slot alternates. That is the DBO two-microbatch case, so
    #          more than one buffer has to be genuinely usable.
    #   T9div  one rank on a DIFFERENT slot must be wrong. Each slot is its own
    #          symmetric allocation and therefore its own multicast team
    #          (MEASURED: distinct multicast pointers), so a rank on slot 1
    #          reduces over buffers the others never wrote. A bitwise-correct
    #          result here would mean the slot choice does not matter and the
    #          rank-uniform rule is unfounded, so agreement is the failure.
    #          It cannot hang: the pads and the counter are shared and slot
    #          independent, so every barrier still releases.
    # T9div writes rank 0's partial sum into slot `div` on every rank, so the
    # recovery check proves both slots are still usable rather than assuming it.
    if "T9" in TESTS and POOL and NSLOTS >= 2:
        t = 2048 if 2048 in TOKS else sorted(TOKS)[0]
        C, nc = 4, 14
        for slot in range(NSLOTS):
            for rep in range(3):
                bo(t, slot).zero_()
                fuse(t, C, nc, cur, slot=slot)
                torch.cuda.synchronize()
                check("T9uni", t, C, nc, f"slot={slot} rep={rep}", slot=slot)
        for rep in range(6):
            s = rep % NSLOTS
            bo(t, s).zero_()
            fuse(t, C, nc, cur, slot=s)
            torch.cuda.synchronize()
            check("T9uni", t, C, nc, f"alternating slot={s} rep={rep}", slot=s)
        dist.barrier()

        div = 1 if rank == 0 else 0
        for s in range(NSLOTS):
            bo(t, s).zero_()
        fuse(t, C, nc, cur, slot=div)
        torch.cuda.synchronize()
        agreed = bool(torch.equal(refs[t], bo(t, 0)))
        gt = torch.tensor([1.0 if agreed else 0.0], device=DEV)
        dist.all_reduce(gt, op=dist.ReduceOp.MAX)
        checks += 1
        if float(gt) > 0:
            fails.append(("T9div", t, C, nc,
                          "divergent slots still matched the reference",
                          1, 0, 0.0))
            if rank == 0:
                print("FAIL T9div: a rank on a different slot still produced "
                      "the reference answer, so nothing enforces slot "
                      "uniformity and the pool's cross-rank rule is unfounded",
                      flush=True)
        elif rank == 0:
            print("# T9div as required: no rank matched the reference, so every "
                  "rank must select the same slot", flush=True)
        dist.barrier()
        for slot in range(NSLOTS):
            bo(t, slot).zero_()
            fuse(t, C, nc, cur, slot=slot)
            torch.cuda.synchronize()
            check("T9recover", t, C, nc, f"slot={slot}", slot=slot)
        if rank == 0:
            print(f"# T9 done checks={checks} fails={len(fails)}", flush=True)
        dist.barrier()

    n = torch.tensor([float(len(fails)), float(checks), worst[0]],
                     device=DEV)
    dist.all_reduce(n, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"\n# SOAK SUMMARY checks={int(n[1])} distinct_failures={int(n[0])}"
              f" max_abs_diff={float(n[2]):.6g} pool={POOL} slots="
              f"{NSLOTS if POOL else 1} pool_init_ms={pool_init_s*1e3:.1f}"
              f" -> {'ALL PASS' if int(n[0]) == 0 else 'FAILURES PRESENT'}",
              flush=True)
        for f in fails[:20]:
            print("#   ", f, flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
