# Research archive

Earlier experiments and their write-ups. Nothing here is needed to build or run the
final implementation. It is kept because the write-ups record what was measured and
what was rejected, which is otherwise lost.

Machine-specific directories in these files were replaced with `<gpu-host>` (the 8x
H200 box) and `<workstation>` (the development machine). Everything else in them,
including md5 sums, branch names, boot log names and numbers, is unchanged. Scripts
here are archived as-is and will not run without those paths restored.

| Path | What it is |
|---|---|
| `reports/qwen_ep_monokernel_result.md` | The Qwen EP monokernel experiment in full: the ownership-aware kernel, correctness against the fp64 oracle, and the measured result. |
| `reports/ds3_ep_monokernel_result.md` | The same experiment on DeepSeek-V3 EP. This one is a negative result and is why the shipped DeepSeek EP path is not the persistent kernel. |
| `reports/report_s8_ds3.md` | An earlier DeepSeek section: batch and grid work. |
| `kernel_diffs/qwen_ep.diff` | The Qwen EP kernel as a diff against its TP parent. The fastest way to see only what EP changed. |
| `kernel_diffs/ds3_ep.diff` | The same for DeepSeek-V3. |
| `d1_ds3.sh` | The DeepSeek EP monokernel campaign runner used for that experiment. Superseded by `benchmarks/c2_camp.sh`. |
| `ds3_acc.py` | The correctness gate for that same experiment: the DeepSeek EP persistent kernel against an fp64 oracle and a full-width non-EP control, ranks in sequence. It is here and not in `validation/` because it gates the archived kernel, not the shipped DeepSeek EP path. The gate for the shipped path is `validation/ep_silu_gate.py`. |
| `superseded/acc_final.py` | An early single-GPU correctness harness against an fp64 oracle. Superseded by `validation/ep4_acc.py` and `ds3_acc.py` above. Its docstring is worth reading: it records why synthetic weights gave a false failure. |

The two reports are working research logs written while the experiments were running,
not cleaned-up write-ups. They discuss what to present and what to drop, they name
intermediate arms that no result depends on, and their conclusions are what the final
numbers were taken from. They are kept as written.
