# Section 8 material: the validated DeepSeek-V3 standalone results

These are the already validated values from the completed DeepSeek-V3 campaigns, carried
forward unchanged. They are the standalone EP and standalone TP results. No number from
the requested EP+TP combination appears anywhere.

One methodology note applies to the DeepSeek EP system table only. Its Triton baseline
row was measured eager, because at 16384 tokens on DeepSeek-V3 the eager and captured
configurations could not share one `gpu_memory_utilization`, while its Before and After
rows are both captured. The kernel tables and the TP system table have no such split.
This is why the cross model summary compares Before to After for both models, which is
the same definition on both sides, rather than comparing raw baselines that were not
collected under identical conditions.

## 8.1 DeepSeek-V3 EP, system level, prefill TTFT

| Configuration | 8K TTFT | Improvement vs Triton | 16K TTFT | Improvement vs Triton |
|---|---|---|---|---|
| Triton baseline, EP 8, eager | 366.8 ms | reference | 721.9 ms | reference |
| Before optimization, captured | 343.9 ms | +6.24% | 715.4 ms | +1.01% |
| After optimization, captured | 297.0 ms | +18.94% | 625.4 ms | +13.33% |

Before to After: 8K +13.54%, 16K +12.45%. Paired, 5 of 5 boots.

## 8.2 DeepSeek-V3 EP, kernel level, routed MoE layer

| Configuration | 8K us/layer | Improvement vs Triton | 16K us/layer | Improvement vs Triton |
|---|---|---|---|---|
| Triton baseline | 3863.9 | reference | 6854.9 | reference |
| Before optimization | 3536.3 | +8.48% | 6286.9 | +8.29% |
| After optimization | 2971.1 | +23.11% | 5162.8 | +24.68% |

Before to After: 8K +15.98%, 16K +17.87%.

Phase breakdown, us per routed layer, Triton then Before then After:

| Phase | 8K Triton | 8K Before | 8K After | 16K Triton | 16K Before | 16K After |
|---|---|---|---|---|---|---|
| Exposed dispatch wait | 68.8 | 69.7 | 48.2 | 237.0 | 268.0 | 154.8 |
| Routed expert GEMM | 2312.1 | 2038.7 | 1683.7 | 4035.7 | 3542.2 | 2888.3 |
| Exposed combine wait | 999.7 | 931.3 | 737.2 | 1952.2 | 1824.6 | 1437.6 |

## 8.3 DeepSeek-V3 TP, system level, prefill TTFT

| Configuration | 8K TTFT | Improvement vs Triton | 16K TTFT | Improvement vs Triton |
|---|---|---|---|---|
| Triton baseline, TP 8 | 352.08 ms | reference | 734.36 ms | reference |
| Before optimization | 356.38 ms | -1.22% | 736.01 ms | -0.19% |
| After optimization | 343.91 ms | +2.32% | 709.47 ms | +3.38% |

Before to After: 8K +3.50%, 16K +3.57%. Paired, 6 of 6 boots.

## 8.4 DeepSeek-V3 TP, kernel level, routed MoE layer

| Configuration | 8K us/layer | Improvement vs Triton | 16K us/layer | Improvement vs Triton |
|---|---|---|---|---|
| Triton baseline | 3045.6 | reference | 5566.4 | reference |
| Before optimization | 3033.8 | +0.39% | 5558.0 | +0.15% |
| After optimization | 2850.9 | +6.39% | 5175.6 | +7.02% |

Before to After: 8K +6.03%, 16K +6.88%.

Exposed communication, us per routed layer, Triton then Before then After: 8K 524.4,
511.2, 328.3. 16K 964.4, 955.5, 573.1.

## DeepSeek-V3 EP context that does not transfer

DeepSeek-V3 EP ran at EP 8 with prefill context parallel 8 and used the DeepEP high
throughput backend, so it had a real all to all dispatch and a real all to all combine,
and one of its three changes was raising the communication SM count from 20 to 32. Qwen
3.5 runs EP 4 with prefill context parallel 1 on the allgather reduce scatter manager and
has neither collective, so none of that transfers. The reason is source level, not a flag:
`use_all2all_kernels` is `use_ep and (dp_size > 1 or pcp_size > 1 or is_sequence_parallel)`
(`config.py:1057`), which is False on the Qwen topology, so the modular prepare and
finalize path is the no dispatch path.
