#!/usr/bin/env python3
"""Correctness for the DeepSeek-V3 EP=8 persistent monokernel, single GPU, 8 ranks
simulated in sequence, against an fp64 ORACLE and a full-width non-EP control.

WHY SINGLE GPU FIRST. The persistent kernel holds every SM and its software grid
barrier means one wrong rank wedges its peers rather than failing alone, so there is
no safe post-entry fallback in a real 8-rank engine boot. Every rank here runs alone
on one device, in sequence, so a hang or a fault is attributable to exactly one rank
and cannot take a peer down with it.

WHAT AN EP RANK RETURNS. A PARTIAL: the sum over the experts resident on that rank
only, with the router weights ALREADY divided by the sum over all TOP_K GLOBAL slots
(prefill_wgmma_fused_renorm() == 1). Because that denominator is global, every rank
scales its own terms by the identical factor, so the eight partials add to the full
routed MoE output with no further normalization. The engine's own MoERunner performs
this add as its cross-rank all-reduce, because
MoEPrepareAndFinalizeNoDPEPModular.output_is_reduced() is False; here it is a torch
sum, which is the same algebra.

REAL WEIGHTS, REAL ROUTER, SCALE MATCHED ACTIVATIONS. The expert weights, the router
matrix and the router correction bias are the DeepSeek-V3 checkpoint's own tensors,
read from the single safetensors shard that holds the layer. DeepSeek-V3 is 671B and
cannot be forward passed on one GPU, so the hidden states are NOT captured: they are
Gaussians renormalized to unit RMS and then multiplied by the layer's real
post_attention_layernorm.weight, which is exactly the transform the MLP input has
gone through in the model. That gives the real PER CHANNEL scale, which is what fp8
per-128-block activation quantization keys on. This is labelled here and in the
deliverable as scale matched, not captured.

ROUTING REFERENCE. ROUTER_MODE 2 is DeepSeek-V3's grouped sigmoid gate: score =
sigmoid(logit), selection score = score + e_score_correction_bias, group score = sum
of the group's top 2 selection scores over N_GROUP=8 contiguous groups of 32,
TOPK_GROUP=4 groups kept, then top TOP_K=8 experts by selection score, and the WEIGHT
is the UNBIASED sigmoid score renormalized over the eight slots. The reference below
is written from that definition and is compared against the ids and weights the
kernel itself wrote. The checkpoint's routed_scaling_factor of 2.5 appears NOWHERE in
the kernel's router path, so it is applied outside the monokernel by the engine; the
reference and the oracle both omit it, which keeps them consistent with the kernel and
leaves the EP comparison unaffected either way.

GATES, all reported, correctness only, no timing is read here:
  G1 rc == 0 on all eight ranks
  G2 no rank emits a local expert id outside [-1, E_LOCAL-1]
  G3 every (token, slot) pair is owned by EXACTLY one rank
  G4 recovered global routing == the grouped sigmoid reference at the kernel's own
     fp32 precision, AND every token where an fp64 reference disagrees sits under
     float epsilon at one of the two decision boundaries
  G5 the normalized router weights are bit identical across all eight ranks
  G6 coverage 1.0: every output element written (bf16_output is NaN poisoned)
  G7 rel-L2 and cos of the summed partials against the fp64 oracle
  G8 pre-entry decline: the EP build handed a null map returns -120 on both the EP
     entry point and the pre-existing 18 argument ABI, and the non-EP control handed
     a map returns -121
  G9 the recovered global routing and the normalized weights equal what the
     full-width non-EP control computes on the same logits, and the control's own
     distance to the same oracle is reported so the EP change is isolated

THE CONTROL, and why the bf16 rounding floor is NOT the right yardstick. The kernel
quantizes activations to fp8 before the up GEMM, so its distance from an fp64 oracle
is dominated by that quantization and is many times the bf16 output floor no matter
how the experts are partitioned. The number that isolates the EP change is a
FULL-WIDTH NON-EP build of the SAME source at the SAME geometry (E_LOCAL 256,
N_UP 4096, N_HALF 2048, A_ROWS 64, UP_PIPE 3, SHM_TOTAL 218464) holding all 256
experts, scored against the same oracle on the same activations.
"""
import ctypes
import gc
import hashlib
import json
import os
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("ABGPU", "7"))
DEV = "cuda:0"

E, E_LOCAL, TOP_K = 256, 32, 8
K_DIM, N_UP, N_HALF, H_DIM = 7168, 4096, 2048, 7168
N_GROUP, TOPK_GROUP = 8, 4
N_KBLK = K_DIM // 128            # 56, activation scale blocks per token
EP_WORLD = E // E_LOCAL          # 8

# EPSO is the DeepSeek EP build under test. It is NOT one of the three variants
# this repository compiles: reproduce it by applying
# research_archive/kernel_diffs/ds3_ep.diff to kernel/deepseek_tp/src and
# building that copy. CTLSO is the full-width non-EP control, an earlier build of
# the same source at 256 global experts. Both explicit, because the whole point
# of the gate is which binary ran.
SO = os.environ["EPSO"]
CTLSO = os.environ["CTLSO"]
# The local snapshot directory of deepseek-ai/DeepSeek-V3. Required: the gate
# reads real expert weights, the real router matrix and the real correction bias
# off it, so a wrong checkpoint silently changes what is being validated.
CKPT = os.environ["DS3CKPT"]
BS_LIST = [int(x) for x in os.environ.get("BSLIST", "512,2048,8192").split(",")]
LAYERS_UNDER_TEST = [int(x) for x in os.environ.get("LAYERS", "3,30,60").split(",")]
SEED = int(os.environ.get("SEED", "1234"))


def geo_of(cl):
    g = {}
    for n in ("num_experts", "num_experts_local", "ep_capable", "top_k", "k_dim",
              "h_dim", "n_up", "n_half", "block_m", "router_mode", "fused_renorm",
              "shm_total", "grid_size", "n_group", "topk_group", "up_pipe",
              "dn_pipe", "a_stages"):
        try:
            f = getattr(cl, "prefill_wgmma_" + n)
        except AttributeError:
            g[n] = None
            continue
        f.restype = ctypes.c_int
        g[n] = f()
    cl.prefill_wgmma_workspace_bytes.restype = ctypes.c_size_t
    g["workspace_bytes"] = cl.prefill_wgmma_workspace_bytes()
    return g


lib = ctypes.CDLL(SO)
GEO = geo_of(lib)
WSB = GEO["workspace_bytes"]
print(f"artifact under test: {SO}")
print(f"  md5 {hashlib.md5(open(SO, 'rb').read()).hexdigest()}")
print(f"  geometry {json.dumps(GEO)}")
# The harness constants and the binary have to agree before a single byte is
# launched; a mismatch here would silently reinterpret the weight layout.
assert (GEO["num_experts"], GEO["num_experts_local"], GEO["ep_capable"]) == (E, E_LOCAL, 1)
assert (GEO["k_dim"], GEO["h_dim"], GEO["n_up"], GEO["n_half"]) == (K_DIM, H_DIM, N_UP, N_HALF)
assert (GEO["top_k"], GEO["router_mode"], GEO["fused_renorm"]) == (TOP_K, 2, 1)
assert (GEO["n_group"], GEO["topk_group"]) == (N_GROUP, TOPK_GROUP)

FN_EP = lib.launch_prefill_moe_wgmma_q1_ep
FN_EP.restype = ctypes.c_int
FN_ROUTER = lib.launch_prefill_moe_wgmma_q1_router   # pre-existing 18 argument ABI
FN_ROUTER.restype = ctypes.c_int

# THE NON-EP FULL-WIDTH CONTROL. Every real build exports the same canonical symbol
# names, so it must be bound through its OWN dlopen handle or both resolve into
# whichever library loaded first (acc_final.py's own finding).
ctl = ctypes.CDLL(CTLSO)
CGEO = geo_of(ctl)
FN_CTL = ctl.launch_prefill_moe_wgmma_q1_ep
FN_CTL.restype = ctypes.c_int
CWSB = CGEO["workspace_bytes"]
print(f"control: {CTLSO}")
print(f"  md5 {hashlib.md5(open(CTLSO, 'rb').read()).hexdigest()}")
print(f"  geometry {json.dumps(CGEO)}")
assert (CGEO["num_experts_local"], CGEO["ep_capable"]) == (E, 0)
# The control is only a control if the ONLY thing that differs is the expert count.
for k in ("num_experts", "top_k", "k_dim", "h_dim", "n_up", "n_half", "block_m",
          "router_mode", "fused_renorm", "shm_total", "grid_size", "up_pipe",
          "dn_pipe", "a_stages"):
    assert GEO[k] == CGEO[k], (k, GEO[k], CGEO[k])
addrs = [ctypes.cast(FN_EP, ctypes.c_void_p).value,
         ctypes.cast(FN_CTL, ctypes.c_void_p).value]
assert len(set(addrs)) == len(addrs), f"symbol aliasing across handles! {addrs}"

IDX = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]


def load_layer(li):
    """One MoE layer's real tensors, built into the kernel's own weight layout.

    w13 is ROW BLOCK CONCATENATED, gate rows [0, N_HALF) then up rows
    [N_HALF, N_UP), which is what the up phase reads (is_up selects feature pass
    GATE_PASSES + g) and what vLLM's own w13_weight holds. The launcher's
    "pair-interleaved" comment on that argument is stale and is not the code.

    MEASURED: one layer's 256 experts span three to four safetensors shards, and the
    layernorm weight lands in yet another, so the index is inverted to shard -> keys
    and each shard is opened exactly once. A loader that assumed a single shard would
    KeyError on the first expert it did not find, which is the safe failure, but the
    inversion also keeps the read sequential over the file.
    """
    p = f"model.layers.{li}.mlp"
    need = {}
    for e in range(E):
        for n, kd in (("gate_proj", "g"), ("up_proj", "u"), ("down_proj", "d")):
            for suf, ks in ((".weight", "w"), (".weight_scale_inv", "s")):
                key = f"{p}.experts.{e}.{n}{suf}"
                need.setdefault(IDX[key], []).append((key, kd + ks, e))
    for key, tag in ((f"{p}.gate.weight", "gate"),
                     (f"{p}.gate.e_score_correction_bias", "bias"),
                     (f"model.layers.{li}.post_attention_layernorm.weight", "ln")):
        need.setdefault(IDX[key], []).append((key, tag, -1))
    w13 = torch.empty((E, N_UP, K_DIM), device=DEV, dtype=torch.float8_e4m3fn)
    w2 = torch.empty((E, H_DIM, N_HALF), device=DEV, dtype=torch.float8_e4m3fn)
    w13s = torch.empty((E, N_UP // 128, N_KBLK), device=DEV, dtype=torch.float32)
    w2s = torch.empty((E, H_DIM // 128, N_HALF // 128), device=DEV, dtype=torch.float32)
    got, one = {"gw": 0, "uw": 0, "dw": 0, "gs": 0, "us": 0, "ds": 0}, {}
    HB = N_HALF // 128
    for shard in sorted(need):
        with safe_open(os.path.join(CKPT, shard), framework="pt", device="cpu") as f:
            for key, kd, e in need[shard]:
                t = f.get_tensor(key)
                if kd == "gw":
                    assert t.shape == (N_HALF, K_DIM), (key, t.shape)
                    w13[e, :N_HALF] = t.to(DEV)
                elif kd == "uw":
                    assert t.shape == (N_HALF, K_DIM), (key, t.shape)
                    w13[e, N_HALF:] = t.to(DEV)
                elif kd == "dw":
                    assert t.shape == (H_DIM, N_HALF), (key, t.shape)
                    w2[e] = t.to(DEV)
                elif kd == "gs":
                    assert t.shape == (HB, N_KBLK), (key, t.shape)
                    w13s[e, :HB] = t.float().to(DEV)
                elif kd == "us":
                    assert t.shape == (HB, N_KBLK), (key, t.shape)
                    w13s[e, HB:] = t.float().to(DEV)
                elif kd == "ds":
                    assert t.shape == (H_DIM // 128, HB), (key, t.shape)
                    w2s[e] = t.float().to(DEV)
                else:
                    one[kd] = t.to(DEV)
                    continue
                got[kd] += 1
    assert all(v == E for v in got.values()), got
    gate_w, bias, ln_w = one["gate"], one["bias"].float(), one["ln"]
    assert gate_w.shape == (E, K_DIM), gate_w.shape
    assert bias.shape == (E,), bias.shape
    assert ln_w.shape == (K_DIM,), ln_w.shape
    return dict(w13=w13, w2=w2, w13s=w13s, w2s=w2s, gate=gate_w, bias=bias, ln=ln_w,
                shard=",".join(sorted(need)))


def make_acts(T, ln_w, seed):
    """RMSNorm scale matched activations. NOT captured hidden states.

    DeepSeek-V3 applies post_attention_layernorm before the MLP, so the MLP input is
    RMSNorm(x) * weight. A unit RMS Gaussian times the real weight reproduces the per
    channel scale that fp8 per-128-block activation quantization keys on.
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(T, K_DIM, generator=g, device=DEV, dtype=torch.float32)
    x = x / x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
    return (x * ln_w.float()).bfloat16().contiguous()


def route_ref(logits, bias, dt):
    """DeepSeek-V3's grouped sigmoid gate at precision dt, ids sorted ascending.

    Also returns the two decision-boundary margins, because a reference computed at a
    HIGHER precision than the kernel is not automatically the right yardstick. MEASURED
    on layer 60 at T=8192: an fp64 reference disagrees with the kernel on exactly 1
    token of 8192, at an expert-boundary margin of 7.607e-08, and an fp32 reference
    (the kernel's own precision, sigmoid included) disagrees on 0 of 8192. So the gate
    below compares at fp32 and uses the fp64 pass only to prove that any disagreement
    it finds sits under float epsilon rather than being a routing defect. This is the
    recorded top-k tie trap in float32 form: an exact-match gate against a
    higher-precision reference manufactures a failure with nothing wrong.
    """
    T = logits.shape[0]
    sc = torch.sigmoid(logits.to(dt))
    sel = sc + bias.to(dt)
    EG = E // N_GROUP
    gsc = sel.view(T, N_GROUP, EG).topk(2, dim=-1).values.sum(-1)
    gk = gsc.topk(TOPK_GROUP, dim=-1).indices
    gm = torch.zeros(T, N_GROUP, device=DEV, dtype=torch.bool)
    gm.scatter_(1, gk, True)
    selm = sel.masked_fill(~gm.unsqueeze(-1).expand(T, N_GROUP, EG).reshape(T, E),
                           float("-inf"))
    ids = selm.topk(TOP_K, dim=-1).indices
    w = sc.gather(1, ids)
    w = w / w.sum(-1, keepdim=True).clamp(min=1e-30)
    o = ids.sort(1)
    gs = gsc.sort(1, descending=True).values
    es = selm.sort(1, descending=True).values
    return (o.values, w.gather(1, o.indices),
            (gs[:, TOPK_GROUP - 1] - gs[:, TOPK_GROUP]).abs(),
            (es[:, TOP_K - 1] - es[:, TOP_K]).abs())


def dequant(w_fp8, scale_inv, rows, cols):
    s = scale_inv.to(torch.float64)
    s = s.repeat_interleave(128, 0).repeat_interleave(128, 1)[:rows, :cols]
    return w_fp8.to(torch.float64) * s


def oracle(hs, L, tidx, tw):
    """The full 256-expert routed MoE layer in float64, on the routing the kernel used.

    tidx holds GLOBAL ids recovered from the eight ranks and tw holds the kernel's own
    already-normalized weights, so the only variable between oracle and the summed
    partials is the arithmetic.
    """
    T = hs.shape[0]
    h = hs.to(torch.float64)
    out = torch.zeros(T, H_DIM, device=DEV, dtype=torch.float64)
    for e in torch.unique(tidx):
        e = int(e)
        if e < 0:
            continue
        sel = (tidx == e)
        rows = sel.any(1).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        x = h[rows]
        W13 = dequant(L["w13"][e], L["w13s"][e], N_UP, K_DIM)
        y = x @ W13.T
        gate, up = y[:, :N_HALF], y[:, N_HALF:]
        act = (gate * torch.sigmoid(gate)) * up
        del W13, y, gate, up
        W2 = dequant(L["w2"][e], L["w2s"][e], H_DIM, N_HALF)
        z = act @ W2.T
        w = torch.where(sel[rows], tw[rows].to(torch.float64),
                        torch.zeros((), device=DEV, dtype=torch.float64)
                        ).sum(1, keepdim=True)
        out[rows] += z * w
        del W2, act, z
    return out


def rel_cos(a, b):
    a, b = a.to(torch.float64).flatten(), b.to(torch.float64).flatten()
    return (float((a - b).norm() / b.norm()),
            float(torch.dot(a, b) / (a.norm() * b.norm())))


def build_map(rank):
    """vLLM's own representation, determine_expert_map(): entry g holds the local
    index of global expert g on this rank, or -1 when the expert lives elsewhere.
    """
    m = torch.full((E,), -1, device=DEV, dtype=torch.int32)
    m[rank * E_LOCAL:(rank + 1) * E_LOCAL] = torch.arange(
        E_LOCAL, device=DEV, dtype=torch.int32)
    return m


def _scratch(T, poison):
    z = lambda *s, dt: torch.zeros(*s, device=DEV, dtype=dt)  # noqa: E731
    d = dict(af8=z(T, K_DIM, dt=torch.uint8), asc=z(T, N_KBLK, dt=torch.float32),
             ti=z(T, TOP_K, dt=torch.int32), tw=z(T, TOP_K, dt=torch.float32),
             si=z(T * TOP_K, dt=torch.int32))
    if poison:
        # DELIBERATELY poisoned: output_accum is never memset by the launcher, so a
        # cell that no tile covers holds this pattern. If the Phase 3 ownership skip
        # were missing the poison lands in the output and the gate fails loudly
        # instead of returning a plausible wrong number.
        d["oa"] = torch.full((T * TOP_K, H_DIM), float("nan"), device=DEV,
                             dtype=torch.float32)
        d["bo"] = torch.full((T, H_DIM), float("nan"), device=DEV, dtype=torch.bfloat16)
    else:
        d["oa"] = z(T * TOP_K, H_DIM, dt=torch.float32)
        d["bo"] = torch.full((T, H_DIM), float("nan"), device=DEV, dtype=torch.bfloat16)
    return d


def _args(logits, T, bm, ws, wsb, hs, s, w13, w13s, w2, w2s):
    return (ctypes.c_void_p(logits.data_ptr()), ctypes.c_int(T), ctypes.c_int(bm),
            ctypes.c_void_p(ws.data_ptr()), ctypes.c_size_t(wsb),
            ctypes.c_void_p(hs.data_ptr()),
            ctypes.c_void_p(s["af8"].data_ptr()), ctypes.c_void_p(s["asc"].data_ptr()),
            ctypes.c_void_p(w13.view(torch.uint8).data_ptr()),
            ctypes.c_void_p(w13s.data_ptr()),
            ctypes.c_void_p(w2.view(torch.uint8).data_ptr()),
            ctypes.c_void_p(w2s.data_ptr()),
            ctypes.c_void_p(s["oa"].data_ptr()), ctypes.c_void_p(s["ti"].data_ptr()),
            ctypes.c_void_p(s["tw"].data_ptr()), ctypes.c_void_p(s["si"].data_ptr()),
            ctypes.c_void_p(s["bo"].data_ptr()))


def launch_rank(rank, T, hs, L, logits, emap):
    lo, hi = rank * E_LOCAL, (rank + 1) * E_LOCAL
    w13 = L["w13"][lo:hi].contiguous()
    w2 = L["w2"][lo:hi].contiguous()
    w13s = L["w13s"][lo:hi].contiguous()
    w2s = L["w2s"][lo:hi].contiguous()
    assert w13.shape == (E_LOCAL, N_UP, K_DIM) and w2.shape == (E_LOCAL, H_DIM, N_HALF)
    ws = torch.zeros(WSB, device=DEV, dtype=torch.uint8)
    sc = _scratch(T, poison=True)
    st = torch.cuda.current_stream().cuda_stream
    rc = FN_EP(*_args(logits, T, GEO["block_m"], ws, WSB, hs, sc, w13, w13s, w2, w2s),
               ctypes.c_void_p(L["bias"].data_ptr()),
               ctypes.c_void_p(emap.data_ptr()), ctypes.c_void_p(st))
    torch.cuda.synchronize()
    out = sc["bo"].clone()
    ti, tw = sc["ti"].clone(), sc["tw"].clone()
    del w13, w2, w13s, w2s, ws, sc
    gc.collect()
    torch.cuda.empty_cache()
    return rc, out, ti, tw


def launch_control(T, hs, L, logits):
    """All 256 experts, full width, NO expert map. Same source, same geometry."""
    ws = torch.zeros(CWSB, device=DEV, dtype=torch.uint8)
    sc = _scratch(T, poison=True)
    st = torch.cuda.current_stream().cuda_stream
    rc = FN_CTL(*_args(logits, T, CGEO["block_m"], ws, CWSB, hs, sc,
                       L["w13"], L["w13s"], L["w2"], L["w2s"]),
                ctypes.c_void_p(L["bias"].data_ptr()), None, ctypes.c_void_p(st))
    torch.cuda.synchronize()
    out = sc["bo"].clone()
    ti, tw = sc["ti"].clone(), sc["tw"].clone()
    del ws, sc
    gc.collect()
    torch.cuda.empty_cache()
    return rc, out, ti, tw


ROWS, FAILS = [], []
for li in LAYERS_UNDER_TEST:
    L = load_layer(li)
    print(f"layer {li} loaded from {L['shard']}: w13 {tuple(L['w13'].shape)} "
          f"w2 {tuple(L['w2'].shape)} gate {tuple(L['gate'].shape)} "
          f"bias[|max|]={float(L['bias'].abs().max()):.5f}", flush=True)
    for BS in BS_LIST:
        hs = make_acts(BS, L["ln"], SEED + li)
        logits = F.linear(hs.float(), L["gate"].float()).contiguous()
        maps = [build_map(r) for r in range(EP_WORLD)]
        outs, tis, tws, rcs = [], [], [], []
        for r in range(EP_WORLD):
            rc, bo, ti, tw = launch_rank(r, BS, hs, L, logits, maps[r])
            rcs.append(rc)
            outs.append(bo)
            tis.append(ti)
            tws.append(tw)
            print(f"  L{li:<3} BS={BS:<6} rank {r} rc={rc} "
                  f"local_id_range=[{int(ti.min())},{int(ti.max())}] "
                  f"owned_slots={int((ti >= 0).sum())}", flush=True)

        row = dict(layer=li, BS=BS, shard=L["shard"], rcs=rcs)
        row["g1_all_rc0"] = all(rc == 0 for rc in rcs)
        row["g2_local_in_range"] = all(int(t.min()) >= -1 and int(t.max()) < E_LOCAL
                                       for t in tis)
        owners = torch.stack([(t >= 0).int() for t in tis]).sum(0)
        row["g3_one_owner"] = bool(torch.all(owners == 1))
        row["owned_per_rank"] = [int((t >= 0).sum()) for t in tis]
        gti = torch.full((BS, TOP_K), -1, device=DEV, dtype=torch.int64)
        for r in range(EP_WORLD):
            sel = tis[r] >= 0
            gti = torch.where(sel, tis[r].long() + r * E_LOCAL, gti)
        # G4 compares against the reference at the KERNEL'S OWN fp32 precision. An fp64
        # reference is not a stricter version of the same test, it is a different test:
        # at a decision boundary narrower than float epsilon the two precisions can
        # legitimately resolve the ordering differently with nothing wrong in either.
        # The fp64 pass below therefore does not gate the routing directly. It gates
        # something stronger, that EVERY token where fp64 disagrees sits under float
        # epsilon at one of the two decision boundaries, so a real routing defect (which
        # would land at a margin of order 1e-2, the median group margin) still fails.
        srt = gti.sort(1)
        kw = tws[0].double().gather(1, srt.indices)
        ref_ids, ref_w, _, _ = route_ref(logits, L["bias"], torch.float32)
        row["g4_route_matches_ref"] = bool(torch.equal(srt.values, ref_ids))
        row["g4_weight_maxabs_vs_ref"] = float((kw - ref_w.double()).abs().max())
        row["g4_weights_close"] = bool(torch.allclose(kw, ref_w.double(),
                                                      rtol=1e-5, atol=1e-7))
        ids64, _, gmarg, emarg = route_ref(logits, L["bias"], torch.float64)
        d64 = (ids64 != srt.values).any(1)
        n64 = int(d64.sum())
        row["g4_fp64_disagree_tokens"] = n64
        row["g4_fp64_worst_margin"] = (
            float(torch.minimum(gmarg[d64], emarg[d64]).max()) if n64 else 0.0)
        row["g4_median_group_margin"] = float(gmarg.median())
        row["g4_fp64_all_sub_epsilon"] = bool(row["g4_fp64_worst_margin"] < 1e-5)
        row["g5_weights_identical"] = all(torch.equal(tws[0], tws[r])
                                          for r in range(1, EP_WORLD))
        summed = torch.stack([o.float() for o in outs]).sum(0)
        row["g6_all_finite"] = bool(torch.isfinite(summed).all())
        row["g6_cov"] = float((summed.abs().sum(1) > 0).float().mean())
        with torch.no_grad():
            ref = oracle(hs, L, gti, tws[0])
        r2, cs = rel_cos(summed, ref)
        row["g7_relL2"], row["g7_cos"] = r2, cs
        row["bf16_floor_relL2"] = rel_cos(ref.bfloat16().float(), ref)[0]

        crc, cout, cti, ctw = launch_control(BS, hs, L, logits)
        cr2, ccs = rel_cos(cout.float(), ref)
        row["ctl_rc"] = crc
        row["ctl_relL2"], row["ctl_cos"] = cr2, ccs
        row["ctl_ratio_ep_over_ctl"] = r2 / cr2 if cr2 > 0 else float("inf")
        same_route = bool(torch.equal(cti.long().sort(1).values, gti.sort(1).values))
        same_w = bool(torch.equal(ctw, tws[0]))
        row["ctl_route_same"], row["ctl_weights_bitsame"] = same_route, same_w
        row["g9_control_agrees"] = bool(crc == 0 and same_route and same_w)
        print(f"  L{li:<3} BS={BS:<6} ctl rc={crc} relL2 {cr2:.4e} cos {ccs:.7f}  "
              f"EP/ctl {row['ctl_ratio_ep_over_ctl']:.4f}x  "
              f"route_same={same_route} w_bitsame={same_w}", flush=True)
        ROWS.append(row)
        bad = [k for k in row if k.startswith("g") and row[k] is False]
        if bad:
            FAILS.append((li, BS, bad))
        print(f"  L{li:<3} BS={BS:<6} relL2 {r2:.4e}  cos {cs:.7f}  "
              f"cov {row['g6_cov']:.4f}  bf16 floor {row['bf16_floor_relL2']:.4e}  "
              f"fp64_diff {row['g4_fp64_disagree_tokens']}"
              f"@{row['g4_fp64_worst_margin']:.3e}  "
              f"gates {'PASS' if not bad else 'FAIL ' + ','.join(bad)}", flush=True)
        del outs, tis, tws, summed, ref, hs, logits, cout, cti, ctw
        gc.collect()
        torch.cuda.empty_cache()
    del L
    gc.collect()
    torch.cuda.empty_cache()

# G8 the pre-entry decline. Nothing here may launch a kernel.
T0 = 512
L0 = load_layer(LAYERS_UNDER_TEST[0])
hs0 = make_acts(T0, L0["ln"], SEED)
logits0 = F.linear(hs0.float(), L0["gate"].float()).contiguous()
w13_0 = L0["w13"][:E_LOCAL].contiguous()
w2_0 = L0["w2"][:E_LOCAL].contiguous()
w13s0 = L0["w13s"][:E_LOCAL].contiguous()
w2s0 = L0["w2s"][:E_LOCAL].contiguous()
ws0 = torch.zeros(WSB, device=DEV, dtype=torch.uint8)
sc0 = _scratch(T0, poison=False)
st0 = torch.cuda.current_stream().cuda_stream
common = _args(logits0, T0, GEO["block_m"], ws0, WSB, hs0, sc0,
               w13_0, w13s0, w2_0, w2s0)
rc_null_map = FN_EP(*common, ctypes.c_void_p(L0["bias"].data_ptr()), None,
                    ctypes.c_void_p(st0))
torch.cuda.synchronize()
rc_old_abi = FN_ROUTER(*common, ctypes.c_void_p(L0["bias"].data_ptr()),
                       ctypes.c_void_p(st0))
torch.cuda.synchronize()
cws0 = torch.zeros(CWSB, device=DEV, dtype=torch.uint8)
csc0 = _scratch(T0, poison=False)
m0 = build_map(0)
rc_ctl_map = FN_CTL(*_args(logits0, T0, CGEO["block_m"], cws0, CWSB, hs0, csc0,
                           w13_0, w13s0, w2_0, w2s0),
                    ctypes.c_void_p(L0["bias"].data_ptr()),
                    ctypes.c_void_p(m0.data_ptr()), ctypes.c_void_p(st0))
torch.cuda.synchronize()
g8 = (rc_null_map == -120 and rc_old_abi == -120 and rc_ctl_map == -121)
print(f"\nG8 pre-entry decline: EP build q1_ep with a null map rc={rc_null_map} "
      f"(want -120), EP build pre-existing q1_router ABI rc={rc_old_abi} (want -120), "
      f"non-EP control handed a map rc={rc_ctl_map} (want -121)  "
      f"{'PASS' if g8 else 'FAIL'}")
if not g8:
    FAILS.append(("g8", 0, ["g8_decline_before_entry"]))

out = os.environ.get("OUTJSON", "/tmp/ds3_acc.json")
json.dump(dict(so=SO, ctl=CTLSO, geo=GEO, ctl_geo=CGEO, seed=SEED, rows=ROWS,
               g8=dict(null_map=rc_null_map, old_abi=rc_old_abi,
                       ctl_given_map=rc_ctl_map, pass_=g8),
               fails=FAILS), open(out, "w"), indent=1)
print(f"\n{len(ROWS)} cells, {len(FAILS)} failing -> {out}")
print("VERDICT: " + ("ALL GATES PASS" if not FAILS else f"FAILURES {FAILS}"))
sys.exit(0 if not FAILS else 1)
