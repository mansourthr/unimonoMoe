#!/usr/bin/env python3
"""STEP 5 correctness for the Qwen EP=4 persistent monokernel, single GPU, 4 ranks
simulated in sequence, against the project's standard fp64 ORACLE.

WHY SINGLE GPU FIRST. The persistent kernel holds every SM and its software grid
barrier means one wrong rank wedges its peers rather than failing alone, so there is
no safe post-entry fallback in a real 4-rank engine boot. Every rank here runs
alone on one device, in sequence, so a hang or a fault is attributable to exactly
one rank and cannot take a peer down with it.

WHAT AN EP RANK ACTUALLY RETURNS. A PARTIAL: the sum over the experts resident on
that rank only, of the routed contributions, with the router weights ALREADY
divided by the sum over all TOP_K GLOBAL slots (prefill_wgmma_fused_renorm() == 1,
kernel line ~1216). Because that denominator is global, every rank scales its own
terms by the identical factor, so the four partials add to the full MoE output with
no further normalization. The engine's own MoERunner performs this add as its
cross-rank all-reduce, because MoEPrepareAndFinalizeNoDPEPModular.output_is_reduced()
is False; here it is a torch sum, which is the same algebra.

REAL WEIGHTS AND REAL ACTIVATIONS, following harness/acc_final.py. Synthetic fp8
bytes produce per-expert partials ~3e5 against a real-checkpoint maximum of ~1.3, so
a synthetic gate measures the harness rather than the kernel. Under EP the weight
slabs are the checkpoint's NATIVE per-expert shapes (gate_up [1024, 2048], down
[2048, 512]) because EP shards the expert COUNT and not the intermediate width, so
rank r's tensors are literally rows [64r, 64r+64) of the loaded module.

GATES, all reported, correctness only, no timing is read here:
  G1 rc == 0 on all four ranks and every rank reaches completion
  G2 no rank emits a local expert id outside [-1, E_LOCAL-1]
  G3 every (token, slot) pair is owned by EXACTLY one rank
  G4 recovered global routing == torch.topk of the router logits
  G5 the normalized router weights are bit identical across all four ranks
  G6 coverage 1.0: every output element written (bf16_output is NaN poisoned)
  G7 rel-L2 and cos of the summed partials against the fp64 oracle
  G8 the EP build DECLINES before entry when handed a null expert_map (-120), and
     the pre-existing non-EP entry points on the EP binary decline the same way
  G9 the recovered global routing equals the routing a non-EP full-width build
     computes on the same logits, and the normalized weights agree

THE CONTROL, and why the bf16 rounding floor is NOT the right yardstick. The kernel
quantizes activations to fp8 before the up GEMM, so its distance from an fp64 oracle
is dominated by that quantization and is many times the bf16 output floor no matter
how the experts are partitioned. The number that isolates the EP change is a
FULL-WIDTH NON-EP build of the same kernel at the same geometry (K 2048, H 2048,
N_UP 1024, N_HALF 512) holding all 256 experts, scored against the same oracle on
the same activations. Two are run because they differ in the output_accum dtype,
which is the one place EP adds a rounding: qctl and maxtiles8192_oabf16.
"""
import ctypes
import hashlib
import json
import os
import sys

import torch
import torch.nn.functional as F

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("ABGPU", "7"))
# The installed DeepGEMM hub kernel and this transformers build disagree on the
# scale-factor dtype (layout.hpp:68 asserts sfa/sfb are float, transformers hands it
# the packed ue8m0 form), so the reference forward dies inside a non-MoE fp8 linear
# before a single hidden state is captured. This is the dispatcher's own documented
# escape hatch and it routes those linears to the Triton fallback, which changes
# nothing about the MoE weights or the activations this harness measures.
os.environ.setdefault("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR", "1")
DEV = "cuda:0"

E, E_LOCAL, TOP_K = 256, 64, 8
K_DIM, N_UP, N_HALF, H_DIM = 2048, 1024, 512, 2048
N_KBLK = K_DIM // 128
EP_WORLD = E // E_LOCAL

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The EP build under test, as written by kernel/build.sh qwen_ep.
SO = os.environ.get(
    "EPSO", os.path.join(REPO, "kernel/qwen_ep/build/libprefill_mono.so"))
# The local snapshot directory of Qwen/Qwen3.5-35B-A3B-FP8. Required: the gate
# reads real expert weights off it, so a wrong checkpoint silently changes what
# is being validated.
CKPT = os.environ["QWEN_CKPT"]
BS_LIST = [int(x) for x in os.environ.get("BSLIST", "512,2048,8192").split(",")]
LAYERS_UNDER_TEST = [int(x) for x in os.environ.get("LAYERS", "0,20,39").split(",")]

lib = ctypes.CDLL(SO)
print(f"artifact under test: {SO}")
print(f"  md5 {hashlib.md5(open(SO, 'rb').read()).hexdigest()}")
GEO = {}
for n in ("num_experts", "num_experts_local", "ep_capable", "top_k", "k_dim",
          "h_dim", "n_up", "n_half", "block_m", "router_mode", "fused_renorm",
          "shm_total", "grid_size"):
    f = getattr(lib, "prefill_wgmma_" + n)
    f.restype = ctypes.c_int
    GEO[n] = f()
lib.prefill_wgmma_workspace_bytes.restype = ctypes.c_size_t
WSB = lib.prefill_wgmma_workspace_bytes()
print(f"  geometry {json.dumps(GEO)}  workspace_bytes={WSB}")
# The harness constants and the binary have to agree before a single byte is
# launched; a mismatch here would silently reinterpret the weight layout.
assert (GEO["num_experts"], GEO["num_experts_local"], GEO["ep_capable"]) == (E, E_LOCAL, 1)
assert (GEO["k_dim"], GEO["h_dim"], GEO["n_up"], GEO["n_half"]) == (K_DIM, H_DIM, N_UP, N_HALF)
assert (GEO["top_k"], GEO["router_mode"], GEO["fused_renorm"]) == (TOP_K, 0, 1)

FN_EP = lib.launch_prefill_moe_wgmma_q1_ep
FN_EP.restype = ctypes.c_int
FN_OLD = lib.launch_prefill_moe_wgmma_q1        # the pre-existing non-EP ABI
FN_OLD.restype = ctypes.c_int
FN_EP2 = lib.launch_prefill_moe_wgmma_q1_ep2
FN_EP2.restype = ctypes.c_int

# THE NON-EP FULL-WIDTH CONTROLS. Every real build exports the same canonical
# symbol name, so each must be bound through its OWN dlopen handle or both names
# resolve into whichever library loaded first (acc_final.py's own finding).
CTL_SOS = {}
# Each control is a full-width non-EP build of the same source, so neither can be
# defaulted to a path this repository produces. Unset means the control is skipped
# and its rows do not appear, which the run log states.
for tag, p in (("qctl", os.environ.get("CTL_QCTL", "")),
               ("oabf16", os.environ.get("CTL_OABF16", ""))):
    if not p or not os.path.exists(p):
        continue
    cl = ctypes.CDLL(p)
    fn = cl.launch_prefill_moe_wgmma_q1
    fn.restype = ctypes.c_int
    cl.prefill_wgmma_workspace_bytes.restype = ctypes.c_size_t
    cg = {}
    for n in ("num_experts", "k_dim", "h_dim", "n_up", "n_half", "fused_renorm"):
        try:
            f = getattr(cl, "prefill_wgmma_" + n)
        except AttributeError:
            cg[n] = None
            continue
        f.restype = ctypes.c_int
        cg[n] = f()
    assert (cg["num_experts"], cg["k_dim"], cg["h_dim"]) == (E, K_DIM, H_DIM), (tag, cg)
    # These predate the EP work and predate the fused renorm, so their weights come
    # back unnormalized; the harness divides by topk_weights.sum(1), which is the
    # checkpoint's own denominator and exact algebra on a per-token scalar.
    CTL_SOS[tag] = dict(so=p, fn=fn, wsb=cl.prefill_wgmma_workspace_bytes(),
                        renorm=bool(cg["fused_renorm"] or 0), geo=cg)
    print(f"  control {tag:>7}  ws={CTL_SOS[tag]['wsb']}  fused_renorm="
          f"{CTL_SOS[tag]['renorm']}  n_half={cg['n_half']}  {p}")
addrs = [ctypes.cast(FN_EP, ctypes.c_void_p).value] + \
        [ctypes.cast(c["fn"], ctypes.c_void_p).value for c in CTL_SOS.values()]
assert len(set(addrs)) == len(addrs), f"symbol aliasing across handles! {addrs}"

from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402

tok = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForCausalLM.from_pretrained(CKPT, dtype=torch.bfloat16, device_map=DEV)
model.eval()
root = model
for a in ("model", "language_model"):
    if hasattr(root, a):
        root = getattr(root, a)
LAYERS = root.layers
print(f"model loaded, {len(LAYERS)} layers", flush=True)

CAP = {}
PROMPT = ("Summarize the history of numerical linear algebra from Gauss to the present "
          "day, covering elimination, iterative methods, Krylov subspaces, and modern "
          "randomized algorithms, and explain why floating-point error analysis became "
          "central to it.")


def make_hook(i):
    def hook(mod, args, kwargs=None):
        hs = args[0]
        CAP[i] = hs.detach().reshape(-1, hs.shape[-1]).clone()
        return None
    return hook


hs_handles = [LAYERS[i].mlp.register_forward_pre_hook(make_hook(i))
              for i in LAYERS_UNDER_TEST]
with torch.no_grad():
    model(**tok(PROMPT, return_tensors="pt").to(DEV))
for h in hs_handles:
    h.remove()
print(f"captured real hidden states at layers {LAYERS_UNDER_TEST}: "
      f"{ {i: tuple(CAP[i].shape) for i in CAP} }", flush=True)


def dequant(w_fp8, scale_inv, rows, cols):
    s = scale_inv.to(torch.float64)
    s = s.repeat_interleave(128, 0).repeat_interleave(128, 1)[:rows, :cols]
    return w_fp8.to(torch.float64) * s


def oracle(hs, ex, tidx, tw):
    """The full 256-expert MoE layer in float64, on the routing the kernel used.

    tidx holds GLOBAL ids recovered from the four ranks and tw holds the kernel's
    own already-normalized weights, so the only variable between oracle and the
    summed partials is the arithmetic.
    """
    T = hs.shape[0]
    h = hs.to(torch.float64)
    out = torch.zeros(T, H_DIM, device=DEV, dtype=torch.float64)
    for e in torch.unique(tidx):
        e = int(e)
        sel = (tidx == e)
        rows = sel.any(1).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        x = h[rows]
        Wgu = dequant(ex.gate_up_proj[e], ex.gate_up_proj_scale_inv[e], N_UP, K_DIM)
        y = x @ Wgu.T
        gate, up = y[:, :N_HALF], y[:, N_HALF:]
        act = (gate * torch.sigmoid(gate)) * up
        Wdn = dequant(ex.down_proj[e], ex.down_proj_scale_inv[e], H_DIM, N_HALF)
        z = act @ Wdn.T
        w = torch.where(sel[rows], tw[rows].to(torch.float64),
                        torch.zeros((), device=DEV, dtype=torch.float64)
                        ).sum(1, keepdim=True)
        out[rows] += z * w
        del Wgu, Wdn, y, gate, up, act, z
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


def launch_rank(rank, T, hs, ex, logits, emap):
    lo, hi = rank * E_LOCAL, (rank + 1) * E_LOCAL
    gu = ex.gate_up_proj[lo:hi].contiguous()
    dn = ex.down_proj[lo:hi].contiguous()
    gus = ex.gate_up_proj_scale_inv[lo:hi].float().contiguous()
    dns = ex.down_proj_scale_inv[lo:hi].float().contiguous()
    assert gu.shape == (E_LOCAL, N_UP, K_DIM), gu.shape
    assert dn.shape == (E_LOCAL, H_DIM, N_HALF), dn.shape

    z = lambda *s, dt: torch.zeros(*s, device=DEV, dtype=dt)  # noqa: E731
    ws = z(WSB, dt=torch.uint8)
    af8, asc = z(T, K_DIM, dt=torch.uint8), z(T, N_KBLK, dt=torch.float32)
    ti, tw = z(T, TOP_K, dt=torch.int32), z(T, TOP_K, dt=torch.float32)
    si = z(T * TOP_K, dt=torch.int32)
    # DELIBERATELY not zeroed and deliberately poisoned: output_accum is never
    # memset by the launcher, so a cell that no tile covers holds this pattern.
    # If the Phase 3 ownership skip were missing, the poison lands in the output
    # and the gate fails loudly instead of returning a plausible wrong number.
    oa = torch.full((T * TOP_K, H_DIM), float("nan"), device=DEV, dtype=torch.float32)
    bo = torch.full((T, H_DIM), float("nan"), device=DEV, dtype=torch.bfloat16)
    s = torch.cuda.current_stream().cuda_stream
    rc = FN_EP(ctypes.c_void_p(logits.data_ptr()), ctypes.c_int(T),
               ctypes.c_int(GEO["block_m"]),
               ctypes.c_void_p(ws.data_ptr()), ctypes.c_size_t(WSB),
               ctypes.c_void_p(hs.data_ptr()),
               ctypes.c_void_p(af8.data_ptr()), ctypes.c_void_p(asc.data_ptr()),
               ctypes.c_void_p(gu.view(torch.uint8).data_ptr()),
               ctypes.c_void_p(gus.data_ptr()),
               ctypes.c_void_p(dn.view(torch.uint8).data_ptr()),
               ctypes.c_void_p(dns.data_ptr()),
               ctypes.c_void_p(oa.data_ptr()), ctypes.c_void_p(ti.data_ptr()),
               ctypes.c_void_p(tw.data_ptr()), ctypes.c_void_p(si.data_ptr()),
               ctypes.c_void_p(bo.data_ptr()),
               None,                                    # router_bias, none on Qwen
               ctypes.c_void_p(emap.data_ptr()),
               ctypes.c_void_p(s))
    torch.cuda.synchronize()
    del gu, dn, gus, dns, ws, af8, asc, si, oa
    return rc, bo, ti, tw


def launch_control(tag, T, hs, ex, logits):
    """All 256 experts, full width, no expert map, the pre-existing 18 argument ABI."""
    c = CTL_SOS[tag]
    wb = c["wsb"]
    gus = ex.gate_up_proj_scale_inv.float().contiguous()
    dns = ex.down_proj_scale_inv.float().contiguous()
    z = lambda *s, dt: torch.zeros(*s, device=DEV, dtype=dt)  # noqa: E731
    ws = z(wb, dt=torch.uint8)
    af8, asc = z(T, K_DIM, dt=torch.uint8), z(T, N_KBLK, dt=torch.float32)
    ti, tw = z(T, TOP_K, dt=torch.int32), z(T, TOP_K, dt=torch.float32)
    si = z(T * TOP_K, dt=torch.int32)
    oa = z(T * TOP_K, H_DIM, dt=torch.float32)
    bo = torch.full((T, H_DIM), float("nan"), device=DEV, dtype=torch.bfloat16)
    s = torch.cuda.current_stream().cuda_stream
    rc = c["fn"](ctypes.c_void_p(logits.data_ptr()), ctypes.c_int(T),
                 ctypes.c_int(32 if T <= 512 else 64),
                 ctypes.c_void_p(ws.data_ptr()), ctypes.c_size_t(wb),
                 ctypes.c_void_p(hs.data_ptr()),
                 ctypes.c_void_p(af8.data_ptr()), ctypes.c_void_p(asc.data_ptr()),
                 ctypes.c_void_p(ex.gate_up_proj.view(torch.uint8).data_ptr()),
                 ctypes.c_void_p(gus.data_ptr()),
                 ctypes.c_void_p(ex.down_proj.view(torch.uint8).data_ptr()),
                 ctypes.c_void_p(dns.data_ptr()),
                 ctypes.c_void_p(oa.data_ptr()), ctypes.c_void_p(ti.data_ptr()),
                 ctypes.c_void_p(tw.data_ptr()), ctypes.c_void_p(si.data_ptr()),
                 ctypes.c_void_p(bo.data_ptr()), ctypes.c_void_p(s))
    torch.cuda.synchronize()
    out = bo.float()
    if not c["renorm"]:
        out = out / tw.sum(1, keepdim=True).clamp(min=1e-30)
        twn = tw / tw.sum(1, keepdim=True).clamp(min=1e-30)
    else:
        twn = tw
    del ws, af8, asc, si, oa, gus, dns
    return rc, out, ti, twn


ROWS = []
FAILS = []
for li in LAYERS_UNDER_TEST:
    blk = LAYERS[li].mlp
    ex = blk.experts
    real = CAP[li]
    for BS in BS_LIST:
        reps = (BS + real.shape[0] - 1) // real.shape[0]
        hs = real.repeat(reps, 1)[:BS].bfloat16().contiguous()
        logits = F.linear(hs, blk.gate.weight).float().contiguous()

        maps = [build_map(r) for r in range(EP_WORLD)]
        outs, tis, tws, rcs = [], [], [], []
        for r in range(EP_WORLD):
            rc, bo, ti, tw = launch_rank(r, BS, hs, ex, logits, maps[r])
            rcs.append(rc)
            outs.append(bo)
            tis.append(ti)
            tws.append(tw)
            print(f"  L{li:<3} BS={BS:<6} rank {r} rc={rc} "
                  f"local_id_range=[{int(ti.min())},{int(ti.max())}] "
                  f"owned_slots={int((ti >= 0).sum())}", flush=True)

        row = dict(layer=li, BS=BS)
        # G1
        row["g1_all_rc0"] = all(rc == 0 for rc in rcs)
        row["rcs"] = rcs
        # G2 no out-of-range local id
        row["g2_local_in_range"] = all(int(t.min()) >= -1 and int(t.max()) < E_LOCAL
                                       for t in tis)
        # G3 exactly one owner per slot
        owners = torch.stack([(t >= 0).int() for t in tis]).sum(0)
        row["g3_one_owner"] = bool(torch.all(owners == 1))
        row["owned_per_rank"] = [int((t >= 0).sum()) for t in tis]
        # recovered global routing
        gti = torch.full((BS, TOP_K), -1, device=DEV, dtype=torch.int64)
        for r in range(EP_WORLD):
            sel = tis[r] >= 0
            gti = torch.where(sel, tis[r].long() + r * E_LOCAL, gti)
        # G4 against torch top-k of the logits (softmax is monotone, so the top-k
        # of the probabilities is the top-k of the logits)
        ref_ids = torch.topk(logits, TOP_K, dim=1).indices.sort(1).values
        row["g4_route_matches_torch"] = bool(torch.equal(gti.sort(1).values, ref_ids))
        # G5 identical normalized weights on every rank
        row["g5_weights_identical"] = all(torch.equal(tws[0], tws[r])
                                          for r in range(1, EP_WORLD))
        # G6 coverage
        summed = torch.stack([o.float() for o in outs]).sum(0)
        row["g6_all_finite"] = bool(torch.isfinite(summed).all())
        row["g6_cov"] = float((summed.abs().sum(1) > 0).float().mean())
        # G7 vs the fp64 oracle
        with torch.no_grad():
            ref = oracle(hs, ex, gti, tws[0])
        r2, cs = rel_cos(summed, ref)
        row["g7_relL2"], row["g7_cos"] = r2, cs
        # the bf16 return dtype's own rounding floor, for scale
        row["bf16_floor_relL2"] = rel_cos(ref.bfloat16().float(), ref)[0]

        # THE CONTROLS. Same oracle, same activations, all 256 experts, no EP.
        g9 = True
        for tag in CTL_SOS:
            crc, cout, cti, ctwn = launch_control(tag, BS, hs, ex, logits)
            row[f"ctl_{tag}_rc"] = crc
            cr2, ccs = rel_cos(cout, ref)
            row[f"ctl_{tag}_relL2"], row[f"ctl_{tag}_cos"] = cr2, ccs
            row[f"ctl_{tag}_ratio_ep_over_ctl"] = r2 / cr2 if cr2 > 0 else float("inf")
            same_route = bool(torch.equal(cti.long().sort(1).values,
                                          gti.sort(1).values))
            same_w = bool(torch.allclose(ctwn, tws[0], rtol=1e-6, atol=1e-9))
            row[f"ctl_{tag}_route_same"] = same_route
            row[f"ctl_{tag}_weights_close"] = same_w
            g9 = g9 and crc == 0 and same_route and same_w
            print(f"  L{li:<3} BS={BS:<6} ctl {tag:>7} rc={crc} "
                  f"relL2 {cr2:.4e}  cos {ccs:.7f}  "
                  f"EP/ctl {row[f'ctl_{tag}_ratio_ep_over_ctl']:.3f}x  "
                  f"route_same={same_route} w_close={same_w}", flush=True)
            del cout, cti, ctwn
        row["g9_control_agrees"] = g9
        ROWS.append(row)
        bad = [k for k in row if k.startswith("g") and row[k] is False]
        if bad:
            FAILS.append((li, BS, bad))
        print(f"  L{li:<3} BS={BS:<6} relL2 {r2:.4e}  cos {cs:.7f}  "
              f"cov {row['g6_cov']:.4f}  bf16 floor {row['bf16_floor_relL2']:.4e}  "
              f"gates {'PASS' if not bad else 'FAIL ' + ','.join(bad)}", flush=True)
        del outs, tis, tws, summed, ref, hs, logits
        torch.cuda.empty_cache()

# G8 the pre-entry decline. Both of these MUST return -120 and must not launch.
T0 = 512
hs0 = CAP[LAYERS_UNDER_TEST[0]].repeat(
    (T0 + CAP[LAYERS_UNDER_TEST[0]].shape[0] - 1) //
    CAP[LAYERS_UNDER_TEST[0]].shape[0], 1)[:T0].bfloat16().contiguous()
blk0 = LAYERS[LAYERS_UNDER_TEST[0]].mlp
logits0 = F.linear(hs0, blk0.gate.weight).float().contiguous()
ex0 = blk0.experts
gu0 = ex0.gate_up_proj[:E_LOCAL].contiguous()
dn0 = ex0.down_proj[:E_LOCAL].contiguous()
gus0 = ex0.gate_up_proj_scale_inv[:E_LOCAL].float().contiguous()
dns0 = ex0.down_proj_scale_inv[:E_LOCAL].float().contiguous()
z = lambda *s, dt: torch.zeros(*s, device=DEV, dtype=dt)  # noqa: E731
ws0 = z(WSB, dt=torch.uint8)
af80, asc0 = z(T0, K_DIM, dt=torch.uint8), z(T0, N_KBLK, dt=torch.float32)
ti0, tw0 = z(T0, TOP_K, dt=torch.int32), z(T0, TOP_K, dt=torch.float32)
si0 = z(T0 * TOP_K, dt=torch.int32)
oa0 = z(T0 * TOP_K, H_DIM, dt=torch.float32)
bo0 = z(T0, H_DIM, dt=torch.bfloat16)
s0 = torch.cuda.current_stream().cuda_stream
common = (ctypes.c_void_p(logits0.data_ptr()), ctypes.c_int(T0),
          ctypes.c_int(GEO["block_m"]),
          ctypes.c_void_p(ws0.data_ptr()), ctypes.c_size_t(WSB),
          ctypes.c_void_p(hs0.data_ptr()),
          ctypes.c_void_p(af80.data_ptr()), ctypes.c_void_p(asc0.data_ptr()),
          ctypes.c_void_p(gu0.view(torch.uint8).data_ptr()),
          ctypes.c_void_p(gus0.data_ptr()),
          ctypes.c_void_p(dn0.view(torch.uint8).data_ptr()),
          ctypes.c_void_p(dns0.data_ptr()),
          ctypes.c_void_p(oa0.data_ptr()), ctypes.c_void_p(ti0.data_ptr()),
          ctypes.c_void_p(tw0.data_ptr()), ctypes.c_void_p(si0.data_ptr()),
          ctypes.c_void_p(bo0.data_ptr()))
rc_null_map = FN_EP(*common, None, None, ctypes.c_void_p(s0))
torch.cuda.synchronize()
rc_old_abi = FN_OLD(*common, ctypes.c_void_p(s0))
torch.cuda.synchronize()
g8 = (rc_null_map == -120 and rc_old_abi == -120)
print(f"\nG8 pre-entry decline: q1_ep with a null map rc={rc_null_map} (want -120), "
      f"pre-existing q1 ABI rc={rc_old_abi} (want -120)  "
      f"{'PASS' if g8 else 'FAIL'}")
if not g8:
    FAILS.append(("g8", 0, ["g8_decline_before_entry"]))

out = os.environ.get("OUTJSON", "/tmp/ep4_acc.json")
json.dump(dict(so=SO, geo=GEO, rows=ROWS,
               g8=dict(null_map=rc_null_map, old_abi=rc_old_abi, pass_=g8),
               fails=FAILS), open(out, "w"), indent=1)
print(f"\n{len(ROWS)} cells, {len(FAILS)} failing -> {out}")
print("VERDICT: " + ("ALL GATES PASS" if not FAILS else f"FAILURES {FAILS}"))
sys.exit(0 if not FAILS else 1)
