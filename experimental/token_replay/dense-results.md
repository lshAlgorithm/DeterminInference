# Complete Dense matrix results

The frozen matrix completed36 points and18 paired comparisons: generated/recorded, concurrency1/4/8, KV capacities131072/16384, three repetitions, seed42, and900-second profiling admission per point. All points used the original InferenceX AgentX path on GPU0. Local token runs are nonofficial (`submission_valid=false`).

## Paired effects

Effects below are recorded relative to generated. Each cell has three pairs. Cache effects use common source rows, equally once per row and weighted by prompt tokens; throughput retains the actual closed-loop request mix and drain. Intervals are per-cell95% Student-t repeat intervals, not source-population intervals or simultaneous coverage across all cells.

| KV tokens | Concurrency | Cache difference, percentage points [95% interval] | Throughput change, % [95% interval] |
| ---: | ---: | ---: | ---: |
| 131072 | 1 | +1.6539 [+1.6539, +1.6539] | +0.0882 [+0.0879, +0.0885] |
| 131072 | 4 | +1.6642 [+1.6541, +1.6743] | +0.5689 [+0.2780, +0.8598] |
| 131072 | 8 | +1.6578 [+1.6379, +1.6776] | +0.8466 [+0.5355, +1.1577] |
| 16384 | 1 | +1.6539 [+1.6539, +1.6539] | +0.0881 [+0.0880, +0.0883] |
| 16384 | 4 | +1.1607 [+0.6704, +1.6509] | +0.2510 [-0.7504, +1.2524] |
| 16384 | 8 | -0.0524 [-0.1093, +0.0045] | -0.0237 [-0.2129, +0.1656] |

Identical fixed-seed cache counts give zero observed repeat variance at concurrency1; the degenerate interval does not imply population certainty. Pressure concurrency4 and8 throughput intervals span zero. Pressure concurrency8 cache differences also span zero. Roomy throughput differences remain below1% in their point estimates.

## Request and latency denominators

Each row pools the request count across three runs; speed and latency values are equal-weight means of the three arm-level means. TTFT, inter-token latency (ITL), and per-user output throughput describe the actual request mix, without common-source latency standardization. Per-user throughput is the AgentX output-token-throughput-per-user metric; per-GPU throughput includes scheduling gaps and actual drain.

| KV | c | Mode | Profiling requests | Tokens/s/GPU | Tokens/s/user | TTFT ms | ITL ms |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 131072 | 1 | generated | 654 | 30.233 | 146.641 | 124.798 | 6.930 |
| 131072 | 1 | recorded | 657 | 30.260 | 146.378 | 112.649 | 6.942 |
| 131072 | 4 | generated | 2433 | 110.892 | 135.640 | 121.526 | 7.628 |
| 131072 | 4 | recorded | 2459 | 111.522 | 135.876 | 101.619 | 7.698 |
| 131072 | 8 | generated | 4872 | 219.248 | 118.436 | 134.295 | 8.927 |
| 131072 | 8 | recorded | 4906 | 221.104 | 120.414 | 124.109 | 8.734 |
| 16384 | 1 | generated | 654 | 30.233 | 146.505 | 124.993 | 6.937 |
| 16384 | 1 | recorded | 657 | 30.260 | 146.824 | 113.323 | 6.939 |
| 16384 | 4 | generated | 1821 | 83.545 | 130.353 | 1540.599 | 8.637 |
| 16384 | 4 | recorded | 1824 | 83.755 | 131.241 | 1547.465 | 8.488 |
| 16384 | 8 | generated | 2407 | 106.976 | 106.756 | 4381.323 | 11.827 |
| 16384 | 8 | recorded | 2411 | 106.951 | 107.567 | 4377.468 | 11.800 |

## Cache work and sampled residency

Counts are totals across the three profiling runs. Evictions count token eviction events, so a token can contribute more than once. Retractions count request retractions; they are not a measure of recomputed GPU FLOPs. Response-usage new prefill is prompt tokens minus cached tokens and does not separately account for internal retraction work. Residency is the maximum of five-second samples across the three runs.

| KV | c | Mode | Input tokens | Cached tokens | New prefill | Evicted tokens | Retractions | Resident max | Allocated KV max |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 131072 | 1 | generated | 4193637 | 3620025 | 573612 | 303167 | 0 | 131059 | 131062 |
| 131072 | 1 | recorded | 4229961 | 3725586 | 504375 | 222694 | 0 | 131066 | 131072 |
| 131072 | 4 | generated | 13709235 | 12052484 | 1656751 | 1721203 | 0 | 131072 | 131072 |
| 131072 | 4 | recorded | 13867114 | 12428144 | 1438970 | 1508231 | 0 | 131055 | 131069 |
| 131072 | 8 | generated | 27048058 | 23829790 | 3218268 | 3810531 | 0 | 131063 | 131072 |
| 131072 | 8 | recorded | 27289700 | 24503516 | 2786184 | 3382142 | 0 | 131059 | 131072 |
| 16384 | 1 | generated | 4193637 | 3620025 | 573612 | 639216 | 0 | 16361 | 16364 |
| 16384 | 1 | recorded | 4229961 | 3725586 | 504375 | 573777 | 0 | 16366 | 16384 |
| 16384 | 4 | generated | 10881105 | 4002515 | 6878590 | 7102297 | 0 | 16344 | 16384 |
| 16384 | 4 | recorded | 10898038 | 4122155 | 6775883 | 7001962 | 3 | 16360 | 16384 |
| 16384 | 8 | generated | 13208587 | 1332659 | 11875928 | 12205577 | 6 | 16352 | 16384 |
| 16384 | 8 | recorded | 13239843 | 1328698 | 11911145 | 12233468 | 8 | 16292 | 16384 |

## Phase costs

| Phase, total across36 points | Requests | Errors/skips/cancellations | Seconds including drain | Input tokens | Cached tokens | New prefill | Output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| warmup | 1716 | 0 | 968.132 | 13660163 | 6458573 | 7201590 | 1716 |
| profiling | 25755 | 0 | 32485.970 | 146988876 | 98291183 | 48697693 | 3154752 |

Server initialization ranged from25.031 to51.067s per run. GPU0 sampled memory maxima ranged from7426 to13772MiB. The maximum across repetitions matches between modes within every capacity/concurrency cell; one-MiB, five-second measurements cannot establish zero incremental replay VRAM. The separate allocator probe measures the replay component directly.

All37 audited runs (36 Dense plus the public Weka reference) completed their warmup HTTP requests before the first profiling request. Boundary gaps were9.430–104.740ms. This is a client request-boundary observation, not a GPU event timeline or evidence that warmup has no cost.

## Evidence

- [Full phase CSV,72 rows](evidence/dense-formal-01.csv)
- [Phase and source-level JSON](evidence/dense-formal-01-summary.json)
- [Paired comparison](evidence/dense-formal-01-comparison.json)
- [Phase-boundary audit](evidence/phase-boundary-final.json)
- [Meeting report and correctness evidence](../../inferencex-e2e/docs/token-replay-meeting-review.md)
