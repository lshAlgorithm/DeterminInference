# Interim findings: 27 completed Dense points

This snapshot contains 27 of the 36 frozen 900-second points, 13 complete
generated/recorded pairs, 18427 profiling requests and 1210 warmup requests.
All completed requests pass source-length and cached-count checks, with zero
errors/skips/cancellations. This is not a token-ID audit of the performance runs.
The independent all447-request exact-ID reference and final study review remain
pending. One completed unpaired arm contributes to the request totals but not
the paired effects below.

## Cache and performance observations

The main comparison uses identical target-tokenized Exgentic prompts, output
lengths, original AgentX scheduling and warmup, seed42 and cache configuration.
Recorded mode replaces sampled IDs before the next normal forward. Cache effects
are standardized to source rows present in both arms, equally once per source
row and weighted by prompt tokens. Throughput retains the actual closed-loop
request mix and drain denominator.

| KV capacity | Concurrency | Complete pairs | Mean standardized cache change (percentage points) | Mean output throughput change |
| --- | ---: | ---: | ---: | ---: |
| 131072 | 1 | 3 | +1.654 | +0.088% |
| 131072 | 4 | 2 | +1.663 | +0.537% |
| 131072 | 8 | 2 | +1.655 | +0.919% |
| 16384 | 1 | 2 | +1.654 | +0.088% |
| 16384 | 4 | 2 | +1.104 | +0.126% |
| 16384 | 8 | 2 | -0.065 | -0.068% |

The roughly1.65-percentage-point cache difference repeats across all roomy
conditions. The tested generated-token drift therefore changes cache reuse;
the observed whole-workload throughput difference is under1%. This supports a
benchmark-fidelity motivation for replay, rather than a general acceleration claim.

Roomy actual-mix mean TTFT is124.80 to112.65ms at concurrency1,121.07 to101.98ms
at concurrency4, and136.50 to123.71ms at concurrency8. These are equally weighted
means of arm-level means over the available pairs. They are descriptive actual-
mix latency differences, not source-standardized latency effects.

At pressure concurrency4 the two throughput effects change sign (-0.214%,
+0.467%). At pressure concurrency8 both are near zero (-0.069%, -0.066%).
The first pressure repeat contains actual evictions and retractions. Replay
does not establish a remedy for capacity pressure or a general reduction in
recomputation. Cached/new-prefill counts are not GPU FLOPs or execution time.

Only roomy concurrency1 has all three pairs here; its throughput95% t interval
is[+0.08792%, +0.08853%]. Identical fixed-seed cached counts give a degenerate
repeat-level cache interval. Neither interval represents source-population
uncertainty. The other cells remain descriptive with two pairs and no final
interval. Final interpretation waits for all36 points, including negative results.

## Completed complementary evidence

The fixed32-prompt matched-output greedy HTTP overhead experiment has upper
adjusted interval bounds below1% at concurrency1/4/8, meeting the scoped3%
replacement-cost target. This is not a universal full-AgentX overhead guarantee.

The full308-request MoE real/synthetic comparison establishes different routed
token distributions under fixed TP2/EP2 placement. Recorder-off throughput
differs by+0.2336%,95% interval[-0.0285%, +0.4958%], so a throughput separation
is not established. Recorder itself adds about21% service time and is excluded
from content-performance measurements. Expert counts do not establish GPU-time
causality.

The original public Weka warmup recreates snapshot cache state and then executes
accelerated one-token trajectory requests. Its44 warmup requests required
1349059 new prefill tokens and340.261s, demonstrating material cache preparation,
separate from model initialization and the900-second profiling denominator.

## Frozen evidence

- [27-point phase summary](evidence/dense-interim-27-summary.json)
- [13 paired effects and denominators](evidence/dense-interim-27-comparison.json)
- [Source-length and cached-count audit](evidence/dense-interim-27-usage-audit.json)
- [Full protocol, completed controls and claim boundaries](../../inferencex-e2e/docs/token-replay-meeting-review.md)

This interim snapshot has deterministic usage checks. It has not received a new
independent final review. Prior Codex reviews remain provisional; Claude review
is unavailable and no dual-review acceptance is claimed.
