import ctypes
import sys

if len(sys.argv) < 2:
    raise SystemExit("usage: geo_readback.py <path to libprefill_mono.so>")
so = sys.argv[1]
lib = ctypes.CDLL(so)
names = ["num_experts", "num_experts_local", "ep_capable", "top_k", "max_tiles",
         "grid_size", "k_dim", "h_dim", "n_up", "n_half", "block_m",
         "router_mode", "fused_renorm", "shm_total", "up_pipe", "dn_pipe",
         "a_stages", "tp_max_c", "tp_has_residual", "tp_has_devseq"]
print(so)
for n in names:
    try:
        f = getattr(lib, "prefill_wgmma_" + n)
    except AttributeError:
        print("%-20s ABSENT" % n)
        continue
    f.restype = ctypes.c_int
    print("%-20s %d" % (n, f()))
f = lib.prefill_wgmma_workspace_bytes
f.restype = ctypes.c_size_t
print("%-20s %d" % ("workspace_bytes", f()))
for e in ("launch_prefill_moe_wgmma_q1_ep", "launch_prefill_moe_wgmma_q1_ep2"):
    print("%-20s %s" % (e.replace("launch_prefill_moe_wgmma_", ""),
                        "present" if hasattr(lib, e) else "ABSENT"))
