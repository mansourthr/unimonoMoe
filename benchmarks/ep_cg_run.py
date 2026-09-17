"""Real vLLM serving boot for the EP CUDA-graph validation, with capture/replay EVIDENCE.

WHY THIS IS A SEPARATE RUNNER. Every shipping EP number (8192 316.1 ms, 16384 631.0 ms) was
produced by the plain eager-mode runner, which was therefore frozen rather than edited. This
file is derived from it and adds exactly three things:

  1. CGSIZES, so `cudagraph_capture_sizes` can be set to include the BENCHMARK shape.
  2. a post-boot evidence readback from every rank (mode, capture list, captured token
     counts, `num_cudagraph_captured`, backend).
  3. an optional probe (CGPROBE=1) that counts real `CUDAGraph.replay()` calls and records
     what `cudagraph_manager.dispatch` actually returned for the timed forward.

⚠️⚠️ WITHOUT (1) A "GRAPH" BOOT SILENTLY RUNS EAGER AT THE BENCHMARK SHAPE. vllm.py:1918
defaults `max_cudagraph_capture_size` to 512 (1024 on Blackwell), and the auto size list is
`[1,2,4] + range(8,256,8) + range(256,max+1,16)` = 51 entries topping out at 512. An
8192-token prefill then misses every candidate and `CudaGraphManager.dispatch` returns
`cg_mode=NONE` (cudagraph_utils.py:407) with no warning. Proving `cudagraph_mode=PIECEWISE`
is NOT proving the measured shape replayed a graph, which is why (2) and (3) exist.

⚠️ THE CAPTURE SIZE IS THE GLOBAL TOKEN COUNT, NOT THE PER-RANK SHARD. Under PCP the
descriptor is chosen in `dispatch_cg_and_sync_dp` before `maybe_partition_pcp_batch` runs
(model_runner.py:1109 sets num_tokens_after_padding from batch_desc, the PCP split happens at
line 1301), so a T=8192 request needs 8192 in the list even though each rank then computes
1024 tokens.

⚠️ An explicitly supplied size list is filtered by `i <= max_num_batched_tokens`
(vllm.py:1944), and this harness sets `max_num_batched_tokens=TOKENS`, so TOKENS itself is the
largest legal capture size. Anything above it is dropped SILENTLY.

⚠️ CGPROBE adds host work to the timed path, so it is for evidence boots only. Timing
campaigns must run with CGPROBE unset.

★★ CGCUSTOM EXISTS BECAUSE A GRAPH BOOT SILENTLY SWAPS THE ACTIVATION KERNEL, AND THAT SWAP
CRASHES CAPTURE. vllm.py:1376-1382 appends the base mode `none` to `custom_ops` whenever the
backend is inductor and the mode is not NONE, and `all` otherwise. So an eager boot runs
`SiluAndMul.forward_cuda` (torch.ops._C.silu_and_mul) while a graph boot runs
`forward_native` wrapped in `CustomOp.maybe_compile(..., dynamic=False)`. Two consequences:

  1. CORRECTNESS OF THE COMPARISON. The eager anchors and any graph boot would differ in the
     shared-expert and dense-MLP activation kernel, which is a second difference on top of
     graph replay. `+silu_and_mul` removes it and makes both arms use the C kernel.
  2. THE ACTUAL BOOT BLOCKER. MEASURED on the byte-stock tree, AgRs, T=8192, PIECEWISE,
     CGSIZES=8192: the 8192 prefill graph captures 1/1 and then `warmup_kernels` decode step
     (warmup.py:379) captures a decode descriptor, and inside that capture the shared expert
     reaches a NEW shape, so `maybe_compile`'s dynamic=False wrapper recompiles
     `triton_poi_fused_mul_silu_slice_0`, and Inductor's pointwise autotuner calls
     `torch.cuda.synchronize()` (benchmarking.py:376). That raises
     `cudaErrorStreamCaptureUnsupported` ("operation not permitted when stream is
     capturing"), and the boot then dies with the SECONDARY
     `cudaErrorStreamCaptureInvalidated`. The primary error is the autotune sync, NOT the
     all2all backend.

⚠️ THE INVALIDATED-CAPTURE ERROR IS A SYMPTOM TWICE OVER. `cudaErrorStreamCaptureInvalidated`
is what the driver reports for every subsequent call once any capture has been poisoned, so it
names neither the offending op nor the backend. Always read upstream to the FIRST exception.

env:
  A2A       all2all_backend, e.g. deepep_high_throughput / allgather_reducescatter
  TOKENS    prompt length; also max_num_batched_tokens
  REPS      generate() calls; rep 0 is warmup and is never the reported number
  EAGER     1 = enforce_eager, 0 = compiled + CGMODE
  CGMODE    NONE / PIECEWISE / FULL...  (PCP accepts PIECEWISE only, pcp_manager.py:161)
  CGSIZES   comma separated capture sizes; default TOKENS when EAGER=0
  CGCUSTOM  comma separated custom_ops entries, default "+silu_and_mul"; "-" disables the
            default and leaves vLLM's own resolution untouched
  CGIND     comma separated k=v inductor_compile_config entries, e.g.
            "triton.autotune_pointwise=False". A blunter lever than CGCUSTOM: it stops the
            autotuner benchmarking ANY pointwise kernel, so it also covers a second op if one
            turns up. Values true/false/int are converted, everything else stays a string.
  CGPROBE   1 = install replay + dispatch probes (evidence boot only)
  CGSPLIT   comma separated extra ops to APPEND to cudagraph splitting_ops, e.g.
            "vllm::moe_forward_shared". A piecewise graph is cut at every splitting op, so an
            op named here runs OUTSIDE the captured region. This is the second candidate graph
            fix for DeepEP HT: instead of removing the dispatch's host sync with
            num_worst_tokens, leave the sync alone and put the whole MoE outside the graph.
            ⚠️ The default is `CompilationConfig._attention_ops` and it is read from the class
            at runtime rather than hardcoded, because passing an explicit list suppresses
            set_splitting_ops_for_v1's own default (compilation.py:1156) and dropping the
            attention ops would silently change what is captured.
  OUTTOK    generated tokens, default 1
  OUTLENS   decode ladder, e.g. "1 4 8 16". Unset = the single OUTTOK width, unchanged.
            Every width runs REPS times inside ONE process, because the process is the unit of
            variance here and a slope assembled across boots would carry boot drift.
            ⚠️ TPOT MUST BE TAKEN FROM A LATER-TOKEN INTERVAL. (D_hi - D_lo) / (hi - lo) with
            lo=1 is anchored on the prefill pedestal: at T=8192 that pedestal is ~350 ms against
            a D16-D8 window of ~70 ms, so a 1% pedestal error moves the slope by ~5%, and with
            lo=1 the leverage is worse still. Use (D16-D8)/8 and cross-check (D8-D4)/4.
"""
import json
import os
import time

# The local snapshot directory of the model under test. Required: an
# accidentally wrong checkpoint is a silent measurement error, not a crash.
MODEL = os.environ["MODEL"]
# vLLM ships this prompt file at benchmarks/sonnet.txt in its own tree.
DATASET = os.environ.get(
    "DATASET",
    os.path.join(os.environ.get("TREE", ""), "benchmarks", "sonnet.txt"),
)
TOKENS = int(os.environ.get("TOKENS", "8192"))
REPS = int(os.environ.get("REPS", "3"))
TP = int(os.environ.get("TP", "1"))
PCP = int(os.environ.get("PCP", "8"))
EP = int(os.environ.get("EP", "1")) == 1
EAGER = int(os.environ.get("EAGER", "1")) == 1
CGMODE = os.environ.get("CGMODE", "PIECEWISE")
CGSIZES = os.environ.get("CGSIZES", "")
CGCUSTOM = os.environ.get("CGCUSTOM", "+silu_and_mul")
CGIND = os.environ.get("CGIND", "")
CGSPLIT = os.environ.get("CGSPLIT", "")
CGPROBE = int(os.environ.get("CGPROBE", "0")) == 1
OUTTOK = int(os.environ.get("OUTTOK", "1"))
# OUTLENS is the decode ladder. Unset means "behave exactly as before", i.e. the single width in
# OUTTOK, so every existing prefill/TTFT campaign reproduces byte for byte. Set it to a list
# ("1 4 8 16") to get a TPOT slope out of one process.
OUTLENS = [int(x) for x in os.environ.get("OUTLENS", "").split()] or [OUTTOK]
BACKEND = os.environ.get("A2A", "")
PLACE = os.environ.get("PLACE", "")
# ADDCFG is a comma separated k=v list forwarded verbatim into LLM(additional_config=...). It exists
# for Qwen3.5, whose Gated DeltaNet linear-attention layers resolve their prefill backend through
# additional_config["gdn_prefill_backend"] and default to a flashinfer path that JIT compiles and
# needs the cutlass python DSL. Unset it and this file behaves exactly as before, so every DeepSeek
# campaign reproduces byte for byte.
ADDCFG = os.environ.get("ADDCFG", "")


def cg_evidence(self):
    """Runs ON each worker. Everything here is a readback, nothing is asserted."""
    import os as _os

    from vllm.compilation.counter import compilation_counter

    cc = self.vllm_config.compilation_config
    mr = getattr(self, "model_runner", None)
    cgm = getattr(mr, "cudagraph_manager", None) if mr is not None else None
    return {
        "rank": self.rank,
        "cudagraph_mode": str(cc.cudagraph_mode),
        "enforce_eager": bool(self.vllm_config.model_config.enforce_eager),
        "capture_sizes_n": len(cc.cudagraph_capture_sizes or []),
        "capture_sizes_max": max(cc.cudagraph_capture_sizes) if cc.cudagraph_capture_sizes else None,
        "max_cudagraph_capture_size": cc.max_cudagraph_capture_size,
        "captured_token_counts_n": len(cgm.captured_token_counts()) if cgm else None,
        "captured_token_counts_max": (max(cgm.captured_token_counts())
                                      if cgm and cgm.captured_token_counts() else None),
        "num_cudagraph_captured": compilation_counter.num_cudagraph_captured,
        # The activation kernel choice is a per-worker readback, not an assumption: an eager
        # boot resolves to base mode `all` and a compiled boot to `none`, so these three
        # fields are what prove the two arms ran the SAME SiluAndMul.
        "custom_ops": list(cc.custom_ops),
        "silu_and_mul_enabled": cc.is_custom_op_enabled("silu_and_mul"),
        "mode": str(cc.mode),
        # Whether the MoE is inside or outside the captured region is a property of THIS list,
        # so it is read back per worker rather than inferred from the launching env. The count
        # plus the moe entries is enough: 14 = tree default (MoE captured), 15/16 = MoE split
        # out. `_attention_ops` is a ClassVar, so a mismatch here would mean the tree changed.
        "splitting_ops_n": len(cc.splitting_ops or []),
        "splitting_ops_moe": [o for o in (cc.splitting_ops or []) if "moe" in o],
        # ⚠️ THE ALLOCATOR IS PART OF THE CONFIGURATION, SO IT IS READ BACK PER WORKER RATHER
        # THAN ASSUMED FROM THE LAUNCHING SHELL. expandable_segments changes the profile-run
        # peak, so an arm that had it and an arm that did not would differ in a second place.
        # This field is what proves every arm of a campaign shared one allocator setting.
        "alloc_conf": _os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "all2all_backend": self.vllm_config.parallel_config.all2all_backend,
        "pcp": self.vllm_config.parallel_config.prefill_context_parallel_size,
        "ep": self.vllm_config.parallel_config.enable_expert_parallel,
    }


def install_probe(self):
    """Count real replays, and record what dispatch returned. Evidence boots only."""
    import torch

    if not getattr(torch.cuda.CUDAGraph, "_ep_probe", False):
        orig_replay = torch.cuda.CUDAGraph.replay
        stats = {"replays": 0}

        def replay(gself):
            stats["replays"] += 1
            return orig_replay(gself)

        torch.cuda.CUDAGraph.replay = replay
        torch.cuda.CUDAGraph._ep_probe = True
        torch.cuda.CUDAGraph._ep_stats = stats

    mr = getattr(self, "model_runner", None)
    cgm = getattr(mr, "cudagraph_manager", None) if mr is not None else None
    if cgm is not None and not getattr(cgm, "_ep_probe", False):
        orig_dispatch = cgm.dispatch

        def dispatch(num_reqs, num_tokens, *a, **k):
            d = orig_dispatch(num_reqs, num_tokens, *a, **k)
            # (tokens asked for, mode granted, tokens the graph was captured at)
            cgm._ep_log.append((num_tokens, str(d.cg_mode).split(".")[-1], d.num_tokens))
            return d

        cgm._ep_log = []
        cgm.dispatch = dispatch
        cgm._ep_probe = True
    return "ok"


def read_probe(self):
    import torch

    stats = getattr(torch.cuda.CUDAGraph, "_ep_stats", None)
    mr = getattr(self, "model_runner", None)
    cgm = getattr(mr, "cudagraph_manager", None) if mr is not None else None
    log = getattr(cgm, "_ep_log", []) if cgm is not None else []
    # ⚠️ `ndisp` AND THE PER-REP CALL EXIST BECAUSE A POOLED REPLAY TOTAL HID THE REAL DEFECT.
    # MEASURED: a 6-rep graph boot reported 310 replays and a 3-rep boot reported 186, and
    # 310 = 62 x 5 while 186 = 62 x 3. So one of the six forwards replayed NOTHING, and it was
    # that forward which returned the wrong token. A tail of four dispatch entries could not
    # show this. Reading this between reps turns "the boot replayed" into "rep i replayed".
    return {"rank": self.rank,
            "replays": (stats or {}).get("replays"),
            "ndisp": len(log),
            "dispatch_tail": log[-4:]}


def report_backend(llm):
    pc = llm.llm_engine.vllm_config.parallel_config
    cc = llm.llm_engine.vllm_config.compilation_config
    got = getattr(pc, "all2all_backend", None)
    print("[cg] APPLIED all2all_backend=%r use_all2all=%s ep=%s pcp=%s tp=%s"
          % (got, getattr(pc, "use_all2all_kernels", None), pc.enable_expert_parallel,
             pc.prefill_context_parallel_size, pc.tensor_parallel_size), flush=True)
    if BACKEND and got != BACKEND:
        print("[cg] FALLBACK: asked for %r, got %r" % (BACKEND, got), flush=True)
    print("[cg] APPLIED enforce_eager=%s cudagraph_mode=%s capture_sizes=%s max_capture=%s"
          % (llm.llm_engine.vllm_config.model_config.enforce_eager, cc.cudagraph_mode,
             cc.cudagraph_capture_sizes, cc.max_cudagraph_capture_size), flush=True)


def main():
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    # ⚠️ Byte identical to ep_pcp_run.py's build_prompt, including add_special_tokens=False.
    # A different tokenization would be a different workload and would void the comparison
    # against the shipping 316.1 / 631.0 ms eager anchors.
    ids = tok(open(DATASET).read(), add_special_tokens=False)["input_ids"]
    while len(ids) < TOKENS:
        ids = ids + ids
    ids = ids[:TOKENS]

    kwargs = {}
    if BACKEND:
        kwargs["all2all_backend"] = BACKEND
    if PLACE:
        kwargs["expert_placement_strategy"] = PLACE
    if ADDCFG:
        kwargs["additional_config"] = {
            k.strip(): v.strip()
            for k, v in (e.split("=", 1) for e in ADDCFG.split(",") if e.strip())}
        print("[cg] ADDCFG %s" % json.dumps(kwargs["additional_config"], sort_keys=True), flush=True)
    if not EAGER:
        from vllm.config import CompilationConfig
        sizes = [int(s) for s in CGSIZES.split(",") if s.strip()] or [TOKENS]
        ckw = {}
        # ⚠️ Do NOT pass a base mode here. vllm.py:1376 appends `none` or `all` itself and
        # is_custom_op_enabled raises if it sees two, so only the +/- directives go in.
        cops = [s.strip() for s in CGCUSTOM.split(",") if s.strip() and s.strip() != "-"]
        if cops:
            ckw["custom_ops"] = cops
        if CGIND:
            def _v(s):
                if s.lower() in ("true", "false"):
                    return s.lower() == "true"
                try:
                    return int(s)
                except ValueError:
                    return s
            ckw["inductor_compile_config"] = {
                k.strip(): _v(v.strip())
                for k, v in (e.split("=", 1) for e in CGIND.split(",") if e.strip())}
        if CGSPLIT:
            # ⚠️ THE ATTENTION OPS ARE RE-ADDED BY HAND BECAUSE AN EXPLICIT LIST REPLACES THE
            # DEFAULT RATHER THAN EXTENDING IT. set_splitting_ops_for_v1 (compilation.py:1156)
            # assigns `list(self._attention_ops)` only when splitting_ops is None, so passing
            # ["vllm::moe_forward_shared"] alone would put attention back INSIDE the graph and
            # change two things at once. Reading the class attribute keeps this list in sync
            # with the tree instead of freezing a 14-entry copy of it here.
            ckw["splitting_ops"] = list(CompilationConfig._attention_ops) + [
                s.strip() for s in CGSPLIT.split(",") if s.strip()]
        kwargs["compilation_config"] = CompilationConfig(
            cudagraph_mode=CGMODE, cudagraph_capture_sizes=sizes, **ckw)
    llm = LLM(
        model=MODEL, trust_remote_code=True,
        tensor_parallel_size=TP, prefill_context_parallel_size=PCP,
        enable_expert_parallel=EP, enforce_eager=EAGER,
        max_model_len=TOKENS + 64, max_num_batched_tokens=TOKENS,
        gpu_memory_utilization=float(os.environ.get("GPUUTIL", "0.92")),
        enable_prefix_caching=False, seed=0, **kwargs,
    )
    report_backend(llm)
    for ev in llm.collective_rpc(cg_evidence):
        print("[cg] EVIDENCE %s" % json.dumps(ev, sort_keys=True), flush=True)
    if CGPROBE:
        print("[cg] PROBE install=%s" % llm.collective_rpc(install_probe), flush=True)

    # DECODE SUPPORT. One boot, several output widths, so a TPOT slope is measured inside ONE
    # process. The process is the unit of variance in this family, so taking D16 from one boot
    # and D8 from another would put between-boot drift straight into the slope.
    #
    # ⚠️ THE LEGACY `rep=` LINE IS EMITTED ONLY FOR outlen == 1, and unchanged. read_camp.py's
    # regex ends in `(.*)$`, so appending a field to that line would be swallowed into
    # first_token. The multi-width path therefore uses its own `drep=` line, and a decode boot
    # stays readable as a TTFT boot by the existing reader at no cost.
    for D in OUTLENS:
        assert D <= 64, "max_model_len is TOKENS+64, so outlen %d has no room" % D
        sp = SamplingParams(max_tokens=D, temperature=0.0)
        for r in range(REPS):
            t0 = time.perf_counter()
            out = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp)
            dt = (time.perf_counter() - t0) * 1e3
            ntok = len(out[0].outputs[0].token_ids)
            # Readback, not assumed: a width that generated fewer tokens than asked (EOS, or a
            # scheduler cap) is not the width it claims and would flatten the slope.
            print("[cg] drep=%d outlen=%d gen=%d wall=%.1f ms" % (r, D, ntok, dt), flush=True)
            # CORRECTNESS GATE for the combined EP+TP campaign: the generated token IDS, not the
            # decoded text. Two arms can print an identical first_token repr while disagreeing on
            # a later id, and a decoded string also hides a tokenizer-level tie at the argmax
            # boundary. Emitted as its own line because read_camp.py's `rep=` regex ends in
            # `(.*)$` and would swallow an appended field into first_token.
            print("[cg] IDS rep=%d outlen=%d ids=%s"
                  % (r, D, list(out[0].outputs[0].token_ids)), flush=True)
            if D == 1:
                print("[cg] rep=%d wall=%.1f ms first_token=%r"
                      % (r, dt, out[0].outputs[0].text), flush=True)
            if CGPROBE:
                # Per-rep replay delta. The dispatch entries added by THIS rep are printed with
                # it, so a rep that was granted cg_mode=NONE is visible as itself rather than
                # inferred from a total. Costs one collective_rpc per rep, which is why CGPROBE
                # boots are never timing boots. `outlen` is on the line because a decode width
                # adds decode-descriptor replays that a prefill-only rep does not.
                pr = llm.collective_rpc(read_probe)
                print("[cg] REPPROBE rep=%d outlen=%d replays=%s new_dispatch=%s"
                      % (r, D, [p["replays"] for p in pr],
                         [p["dispatch_tail"][-1] if p["dispatch_tail"] else None
                          for p in pr][:1]),
                      flush=True)
    if CGPROBE:
        for pr in llm.collective_rpc(read_probe):
            print("[cg] PROBE %s" % json.dumps(pr, sort_keys=True), flush=True)
    print("[cg] done", flush=True)


if __name__ == "__main__":
    main()
