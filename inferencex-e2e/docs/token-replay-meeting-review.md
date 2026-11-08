# Token replay meeting implementation and evidence

The original `agentx profile` route now accepts complete recorded token IDs,
retains the AgentX scheduler, snapshot warmup, spawn/join and session isolation,
and runs through the InferenceX result collector. The full36-point Dense matrix
and the MoE study are complete. A fixed32-prompt HTTP mechanism experiment meets
the scoped3% target. Roomy-cache Dense throughput point estimates improve by
0.088%/0.569%/0.847% at concurrency1/4/8; pressure concurrency4/8 and MoE content
throughput intervals span zero. All447 qualified Dense requests pass the
from-start exact-ID reference. Deterministic delivery checks pass. A fresh
independent Codex review and scoped follow-up found no blocking issues; the
review remains provisional because Claude-review is unavailable.

## Implementation

`AGENTX_TOKEN_TRACE` selects local Weka-shaped token traces and
`AGENTX_REPLAY_MODE=generated|recorded` selects the behavior. Both modes use the
same prompt IDs, output lengths, scheduler and cache configuration. The patched
harness wheel is pinned and verified during bootstrap; the original public-data
route remains the default. Local token aggregates explicitly contain
`submission_valid=false` and the mode, source and seed.

The harness carries paired `input_ids`/`output_ids` through Weka request parsing,
turn construction, snapshots and request dispatch. Full prompt IDs bypass chat
rendering. One-token warmup truncates the replay output to one token too. The
`sglang-token-completions` endpoint supports the existing per-session cache-bust
contract without modifying token IDs: tree members share a `cache_salt`, while
independent tree instances receive different salts. The SGLang scheduler now
forwards that value into `Req.extra_key`; merely sending a salt on the wire was
insufficient before that fix.

SGLang 0.5.10 still executes normal forward and sampling. Its replacement kernel
writes the recorded ID before the next forward and logprob reporting. Replay
storage is packed on device, with metadata-only batch filtering/merging and no
replay state once the last forced request exits. Model embedding lookup remains
normal. Full-trace requests that exceed tokenizer or scheduler admission budgets
are rejected, including the one-token warmup prompt boundary.

The guarantee concerns recorded token trajectories and the tokens represented
by the cache. It does not make floating-point KV tensors bitwise identical
across different batch shapes or replace all engine determinism mechanisms.
Recorded IDs are staged before use; normal embedding lookup remains part of
the model forward. The measured costs below do not identify embedding lookup
as a replay-specific bottleneck that would justify precomputing embeddings.

## Qualified data

Only repository-supported sources are used. Seed is 42.

| Source | Frozen qualification | Interpretation |
| --- | --- | --- |
| Public Weka | Revision `23f152f6f0f9399a85901b89a6458def0ef16729`; complete source downloaded | Hash-only corpus. Original content and reply IDs are unavailable. Separate default-route reproduction only. |
| Exgentic v2 SWE-bench | Revision `4b8ad4ab198438e5a170f9171c19c6a2cf7c1814` | Content is re-tokenized for each target model. IDs are not the source model's original IDs. |
| Dense qualification | 32 sessions, 447 requests; 46 of 1959 candidate sessions examined | Consecutive windows of 4–16 requests; maximum prompt plus output 16143 tokens; no request truncation. Remaining candidates unexamined. |
| MoE qualification | 21 sessions, 308 requests; 128 candidates examined | The older Qwen1.5 template cannot represent tool-bearing requests faithfully, so those spans are excluded. |
| MoE synthetic control | 21 sessions, 308 requests from the repository Shakespeare sampler | Per-session prefix tries preserve input/output lengths, every pairwise prompt/full-response LCP, special-token positions and trace metadata. |

The corrected `exgentic-dense-02` and `exgentic-moe-02` manifests contain source
files, exclusions, selected windows and target template information. Tool-call
argument strings are converted back to JSON objects before Qwen's template
renders them. The older Dense `-01` data double-encoded these arguments in 327
of 447 outputs and is diagnostic-only.

Exgentic timestamps are processed source timing; missing DAG structure is not
invented. The separate MoE service experiment deliberately uses fixed-order,
zero-wait sequential windows and flushes cache between sessions. It does not
claim to reproduce source arrival times or an original DAG.

## Original-path acceptance

Evidence is preserved in Git LFS at
[`entry-evidence-20261106.tar.gz`](../../experimental/token_replay/evidence/entry-evidence-20261106.tar.gz).
The archive includes the frozen datasets, request records, server wire logs,
commands, check receipts and negative evidence.

`dense-dag-02` uses the original InferenceX AgentX command and the bundled Weka
subagent topology, extended as an explicitly synthetic, token-consistent fixture.
It completed 198 profiling requests and 42 warmup requests with zero errors;
the InferenceX collector and validator both completed.

- All 240 actual wire prompts, replay payloads and final output ID sequences
  match the source, with one-token caps only during warmup.
- 52 tree instances have distinct actual wire salts; each tree's parent/children
  share its salt. Event checks cover 44 joins, 48 reset-bearing trees, 40 from-start
  profiling trees, and four warmup-to-profile handoffs.
- 40 consecutive child pairs preserve the expected 0.75-second idle gap; observed
  gaps are approximately 0.751–0.753 seconds. Joined requests follow all observed
  children, with explicit complete-child coverage for from-start trees.
- A separate identical-prompt A/B/A salt witness changes cached-token counts
  from the broken `[0,63,63]` to `[0,0,63]` after the scheduler fix.
- The from-start reference obtains cached-token counts `[0,208,231,208,0]` for
  root, child, child continuation, joined root and disjoint reset respectively.
- Mixed normal/replay requests of lengths 3, 7 and 13 finish correctly, and a
  subsequent ordinary request matches its ordinary reference. Both full-trace
  and warmup-boundary oversize HTTP requests return explicit budget rejections.

The earlier `dense-entry-02` and `dense-dag-01` do not provide valid cache-isolation
or final-entry evidence. Their failures and pre-fix salt leak remain preserved.

## Dense experiment protocol

Capacity preflight used corrected Exgentic data, concurrency 8 and 120-second
profiling admission windows. Both conditions completed with zero request errors.

| KV capacity | Profiling requests | Cache hit fraction from server aggregate | Peak non-evictable usage fraction | Evicted tokens including warmup |
| --- | ---: | ---: | ---: | ---: |
| 131072 | 232 | 0.91162 | 0.33547 | 138234 |
| 16384 | 110 | 0.10060 | 0.99573 | 1202333 |

These numbers qualify two capacities; they are not generated-vs-recorded effects.
“Roomy” means active working sets fit, not zero eviction of stale independent
sessions. The pressure condition demonstrably evicts tokens.

[`dense-formal-01.spec.json`](../../experimental/token_replay/dense-formal-01.spec.json)
freezes 36 points: two modes, concurrency 1/4/8, both capacities, three repeats,
and 900 seconds of profiling admission per point. Paired order alternates.
Every point starts a fresh server after a GPU guard; full request logs are off.
The same GPU0/model/seed/warmup and fixed data apply to both arms.

AgentX remains a closed-loop scheduler: mode-dependent completion times can
change how many requests and which trace mix fit in 900 seconds. Reports retain
actual per-source request counts, phase ranges including drain, errors and
cancellations. Throughput comparisons must retain these denominators. Warmup
and profiling cache/computation metrics are summarized separately. Existing
server eviction and retraction counters are retained; a prefix-cache shortfall
must not automatically be called a measured number of GPU recomputation FLOPs.

The independent mechanism-overhead experiment samples 32 fixed real prompts,
captures their greedy outputs, and compares normal generation against replay of
those exact outputs at concurrency 1/4/8. Each batch starts from a flushed cache.
Every measured batch must match output IDs, prompt lengths and cached-token work.
The timed interval includes HTTP serialization, inference and response receipt;
equal cache flush operations are outside the timer. Both modes are prewarmed;
first replay-use request times are stored separately.

The initial three pairs expand directly to five if the interval crosses 3%.
Bonferroni-adjusted Student t intervals cover the two prespecified looks under
the t assumptions. Acceptance requires the upper bound to be at most 3% at every
measured concurrency. The interval describes repeated execution of these fixed
prompts, not the entire data population or all sampling policies. Retraction
counters must also be zero before claiming identical computation. A completed
measurement with `target_met=false` is a valid negative result.

The completed `dense-overhead-02` measurement had zero retractions, exact output
IDs and matched cached-token work in every batch. All three initial paired
repetitions suffice under the prespecified adjusted intervals:

| Concurrency | Mean additional HTTP wall time | Adjusted interval |
| --- | ---: | ---: |
| 1 | -0.0479% | [-0.9108%, 0.8150%] |
| 4 | -0.0567% | [-0.5019%, 0.3886%] |
| 8 | 0.0109% | [-0.6475%, 0.6693%] |

Negative point estimates are measurement noise, not demonstrated speedups. This
result concerns these fixed32 greedy prompts and the HTTP boundary, not the full
AgentX workload or the data population. Maximum calculated replay payloads were
2536/4120/8192 bytes per issued batch for concurrency1/4/8; those are token and
metadata payload sizes, not measured incremental allocator or peak GPU memory.
Raw results are in [dense-overhead-02.json](../../experimental/token_replay/evidence/dense-overhead-02.json).

The retained storage-maintenance fixture spans batch sizes1/32/128 with mixed
ordinary/replay requests and uneven response lengths. Across its five cases,
synchronized staging took0.379–3.369ms, median metadata filtering10.13–19.07us,
and median merging16.28–18.89us. The fixture packs1024–359424 token-payload bytes;
these sizes exclude metadata and allocator retention. Its separate CUDA-event
kernel probe measured approximately0.98–1.17us per replacement for batch sizes
1–512. These fixture timings explain component costs; only the matched HTTP
experiment above supplies the scoped3% decision. Source, timing protocol and
samples are retained in `profile_replay.py`, `replay-profile-compact.log` and
`replay-kernel-compact.log`.

A further standalone allocator probe used32 actual qualified response sequences,
seed42, on idle GPU2 after the model studies. It measured44 admissions across
concurrency1/4/8, with filtering and merges, and observed:

| Concurrency | Largest token+metadata payload (bytes) | Largest live staged allocation delta (bytes) | Largest allocator peak delta (bytes) |
| --- | ---: | ---: | ---: |
| 1 | 2536 | 3072 | 4096 |
| 4 | 4120 | 4608 | 5632 |
| 8 | 8192 | 8704 | 9728 |

Every admission returned to its allocated-byte baseline after all replay views
were released. Filtering retains the shared packed buffer even when fewer
responses remain. A preallocated index tensor already held a2MiB reserved pool;
zero additional reserved bytes therefore does not mean zero reserved memory.
First CUDA allocation took183.56ms; per-stage synchronized host medians were
226–278us for staging,85–86us for filtering, and84–85us for merging. These single-
operation synchronized timings differ from the20-call amortized fixture above.
First-use costs were separately 0.295ms for staging,
15.012ms for filtering and 5.787ms for merging;
the per-configuration medians above must not substitute for these cold costs.
This is actual component allocator evidence for the selected responses, not
whole-serving incremental VRAM or a population maximum. It does not replace
HTTP end-to-end overhead measurement.

## Complete Dense results

The frozen36-point matrix completed25755 profiling requests and1716 warmup
requests with zero errors, skips or cancellations. The profiling phases
produced3154752 output tokens over32485.970 seconds including drain; the
admission windows total32400 seconds. Every cell has three paired repetitions.

| KV capacity | Concurrency | Standardized cache change, percentage points [95% interval] | Output throughput change, % [95% interval] |
| --- | ---: | ---: | ---: |
| 131072 | 1 | +1.6539 [+1.6539, +1.6539] | +0.0882 [+0.0879, +0.0885] |
| 131072 | 4 | +1.6642 [+1.6541, +1.6743] | +0.5689 [+0.2780, +0.8598] |
| 131072 | 8 | +1.6578 [+1.6379, +1.6776] | +0.8466 [+0.5355, +1.1577] |
| 16384 | 1 | +1.6539 [+1.6539, +1.6539] | +0.0881 [+0.0880, +0.0883] |
| 16384 | 4 | +1.1607 [+0.6704, +1.6509] | +0.2510 [-0.7504, +1.2524] |
| 16384 | 8 | -0.0524 [-0.1093, +0.0045] | -0.0237 [-0.2129, +0.1656] |

Cache standardization uses source rows common to each pair. These are per-cell
Student-t intervals across repeated execution of the fixed seed42 workload,
not source-population intervals or simultaneous coverage across all cells.
Identical fixed-seed cache counts at concurrency1 give zero observed repeat
variance; that does not imply population certainty. Closed-loop timing changes
the actual mix and number of requests, so throughput retains those denominators.

The roomy results support a repeatable cache-reuse effect from generated-token
drift on this workload. Actual-mix mean TTFT changes from124.80 to112.65ms,
121.53 to101.62ms and134.29 to124.11ms at concurrency1/4/8. Those latency values
are descriptive arm-level means, without common-source latency correction.
The corresponding per-GPU output throughput is30.233 to30.260,110.892 to111.522,
and219.248 to221.104 tokens/s. Per-user output throughput and inter-token
latency are retained in the detailed tables; cache changes do not translate
proportionally into overall throughput.

Pressure concurrency8 has about10% cache hits in both modes and no established
cache or throughput separation. Across its three profiling repeats, generated
mode has6 retractions and recorded mode8. Pressure concurrency4 has0 versus3;
all roomy cells have0. Both capacities evict tokens, including stale independent
sessions in the roomy condition. Replay therefore does not establish a general
remedy for capacity pressure or retraction work. Response-usage new prefill
counts do not separately measure internal retraction recomputation or GPU FLOPs.

Dense warmup totals968.132 seconds and7201590 new prefill tokens, separately
from48697693 new prefill tokens during profiling. Server initialization ranges
from25.031 to51.067 seconds per point. Five-second GPU0 memory samples reach
7426–13772MiB depending on capacity and concurrency. The maximum across repeats
matches between modes in each cell; this sampling granularity cannot establish
zero incremental replay memory. The allocator probe above supplies the separate
component measurement.

[Complete Dense tables](../../experimental/token_replay/dense-results.md) include
per-user/per-GPU throughput, TTFT/ITL, actual request counts, cache residency,
evictions, retractions, phase costs and links to the full72-row CSV and source
aggregates. The independent
[from-start reference](../../experimental/token_replay/evidence/dense-corpus-reference.json)
executes all32 qualified sessions and447 requests with exact output ID and
length-finish checks while retaining normal stochastic sampling before replacement.
It is a correctness reference, not an independent performance baseline.
Performance-run source-length checks are separate and do not retrospectively
prove ID equality for every performance repetition.

The [delivery verifier](../../experimental/token_replay/evidence/delivery-verification.json)
passes the exact36-cell grid,900-second conditions, source lengths and counts,
all447 reference outputs and completed MoE, overhead and storage gates. Raw
Dense records, server/client commands, receipts, metrics and the reference run
are in the [Dense evidence archive](../../experimental/token_replay/evidence/dense-formal-evidence-20261106.tar.gz).

## Public Weka reference qualification

The original public corpus has393 traces but no complete trace qualifies at16K:
minimum required context is44048 and maximum996579. `weka-original-01` preserves
that zero-eligible result. A separately labeled64K YaRN reference uses factor2,
original context32768 and rope theta1000000; it is not part of the16K causal
comparison. `weka-original-02` failed on missing rope parameters and `-03` reached
warmup but failed FlashInfer JIT because the isolated CUDA include path lacked
`curand.h`. Both failures are preserved.

A scoped `CPATH` points at existing cuRAND headers and a project-specific
FlashInfer cache isolates the rebuild. A fresh agent executed the documented
seed42 top-k/top-p witness verbatim: actual kernel dispatch, valid sampled IDs
and repeated-seed equality passed. No package upgrade was needed.
`weka-original-04` completed the original public route:7 eligible traces,
4 active trajectories,44 warmup and84 profiling requests, zero errors. The
900-second profiling phase produced44465 output tokens (49.4055 tokens/s),
with2260960/3344115 prompt tokens cached (67.6101%). These are descriptive
numbers for the64K reference, not a cross-dataset improvement baseline. The
Dense36-point matrix started only after this completion receipt.

Original warmup serves two purposes: snapshot primers recreate pre-start cache
state, then accelerated live-trajectory replay adds10 requests per lane with
zero idle delay and one generated token per request. It warms cache pressure
and carries active state into profiling; it does not regenerate each historical
reply. In this public run all44 warmup outputs had length1, but their full
prompts required1349059 new prefill tokens and340.261 seconds, versus1083155
new prefill tokens during profiling. Model load/CUDA-graph initialization was a
separate29.046 seconds. Warmup cost is therefore material and is reported outside
the900-second profiling throughput denominator, not hidden as free preparation.

Five-second GPU0 samples reached13848MiB. Profiling prefix-resident cache sampled
maximum131040 tokens; allocated KV reached131065, including in-flight suffixes.
Those are sampled counts, not instantaneous peaks or replay-specific VRAM costs.

The pinned harness waits for accelerated-warmup wire requests to drain before
profiling handoff. An independent
[request-boundary audit](../../experimental/token_replay/evidence/phase-boundary-final.json)
checks all36 Dense runs and this public reference: every warmup HTTP request
ends before the first profiling request, with gaps of9.430–104.740ms. Thus this
version has no observed overlap of recorded warmup and profiling HTTP requests.
That is not a GPU event timeline, evidence of zero warmup cost, or proof that a
one-token snapshot primer equals complete historical replay. Removing warmup
would change the benchmark's starting-cache contract and is not claimed as a fix.

## MoE experiment protocol

The BF16 Qwen1.5-MoE-A2.7B-Chat weights are fixed at revision
`ec052fda178e241c7c443468d2fa1db6618996be`. GPU2/3 use TP2/EP2, trivial fixed expert
placement, EPLB disabled, and separate shared experts. The isolated environment
and fresh agents following the documented invocations passed the run6 and run7 witnesses.
That witness proves feasibility and exact 32-token replay, not workload effects.

The full service comparison runs all 308 qualified requests in both real and
synthetic arms, three paired repetitions with alternating order. Recorder is
absent from the performance engine. Output IDs are checked exactly on every
request, with request latency, client TTFT and cached-token counts retained.
Service throughput divides output count by summed request wall time; it excludes
flush, file writes and between-session gaps and is not whole-campaign throughput.

The recorder-off full comparison completed all308 requests in each of six arms,
with61931 output and1913941 input tokens per arm. Cached work also matches
per request, totaling1757998 cached tokens per arm. Real-content service
throughput was170.59–171.03 output tokens/s across both GPUs. Synthetic-relative-
to-real paired throughput change was+0.2336%, with a three-pair95% t interval
[-0.0285%, +0.4958%]; this does not establish a throughput difference. Token-
weighted client TPOT changed-0.1275% [-0.1692%, -0.0858%], a small scoped timing
observation that does not identify an expert-routing cause.

Recorder-on runs and per-rank compute/communication diagnostics are separate.
Recorder histograms are global and replicated across ranks: accumulate sessions
within one rank, never sum the two replicas. Token load imbalance is not GPU
execution-time imbalance. The initial first-request profile covers only 1/308
requests and cannot stand in for workload-wide rank timing.

Both complete recorder arms passed durable raw audits:21 sessions/308 requests
per content,42 files per content across both ranks, exact prefill suffix IDs and
positions, exact full-output decode IDs and positions, identity24-by60 placement,
and four routed expert assignments per input per layer. Replicated rank totals
match; aggregation uses rank0 only. Prefill forwards cover155943 input tokens,
and decode forwards cover61931 tokens in each content arm. Under overlap the
last output token is forwarded before its result is discarded, so decode work
here is output length, not output length minus one.

| Phase | Mean layer expert-distribution TV distance, real vs synthetic | Real mean layer card max/mean | Synthetic mean layer card max/mean |
| --- | ---: | ---: | ---: |
| Prefill | 0.16887 | 1.03146 | 1.03593 |
| Decode | 0.17926 | 1.04047 | 1.03643 |

These are routed token distributions under fixed placement, not GPU time.
The three-pair recorder-off throughput interval still spans zero despite the
expert-distribution differences. One recorder-on pass took21.41% longer request
service time for real content and21.30% for synthetic content versus the earlier
three-off mean. These descriptive estimates lack paired drift control;
three additional alternating on/off pairs completed with exact output and
per-request cached-work matches. Their paired95% t estimates are:

| Content | Recorder-on service time increase | Including flush/dump/request writes |
| --- | ---: | ---: |
| Real | 21.1618% [20.5655%, 21.7581%] | 22.5801% [22.2573%, 22.9029%] |
| Synthetic | 21.2044% [20.9994%, 21.4095%] | 22.6846% [22.5092%, 22.8601%] |

Service throughput correspondingly falls17.4655% for real content and17.4947%
for synthetic content. Engine construction takes22.490–24.061 seconds in these
six runs, recorded separately from workload time. Two fixed source requests are
prewarmed per engine before the measured arms. Alternating order reduces order
bias but does not eliminate temporal drift; intervals describe repeats of the
fixed308-request workload. Recorder-on timings are excluded from content
performance comparisons.

The six paired engine runs are retained in the
[recorder control archive](../../experimental/token_replay/evidence/moe-recorder-paired-evidence-20261106.tar.gz).
The [storage and environment archive](../../experimental/token_replay/evidence/storage-doc-witness-evidence-20261106.tar.gz)
contains the qualified allocator probe and final documented environment witness.
Both local archives match the remote SHA-256 values in the
[transfer verification receipt](../../experimental/token_replay/evidence/moe-final-archives-transfer-verification.json).

All9 representative full requests per content completed with their preceding
session history. All36 expected rank trace files were parsed and paired. The
following are sums of within-category GPU interval unions over those9 requests:

| Content | Rank | Compute and other union (s) | Communication union (s) |
| --- | ---: | ---: | ---: |
| Real | 0 | 13.42556 | 2.86225 |
| Real | 1 | 13.66391 | 2.62090 |
| Synthetic | 0 | 13.45985 | 2.84153 |
| Synthetic | 1 | 13.63268 | 2.59123 |

Communication classification matches actual NCCL AllReduce and AllGather kernel
names. Categories overlap across streams and ranks; do not add them as wall
time. These are single-pass profiler-on diagnostics for9/308 selected requests,
with no repeated-profile interval, and cannot establish whole-workload GPU
compute or communication effects. Per-request pairs, kernel names and sums are
retained in [rank comparison](../../experimental/token_replay/evidence/moe-rank-comparison.json)
and the [raw trace/recorder archive](../../experimental/token_replay/evidence/moe-routing-rank-evidence-20261106.tar.gz).

## Review and verification

Fresh Codex reviews found and prompted fixes for target-template argument
encoding, full-trace admission, DAG-check false positives, HTTP budget-check
false positives, cache-flush response parsing and overhead warmup/stopping rules.
The salt-forwarding bug was established by a real failed isolation witness.
Deterministic/live checks support the specific claims above. Codex positive
reviews remain provisional. No `claude-review` MCP tool was available, so no
cross-family review or dual-review acceptance is claimed.

The original-path implementation, short acceptance, full Dense matrix, all-corpus
reference and MoE study are complete. The
[final independent review](../../experimental/token_replay/evidence/final-review-20261107.json)
reconstructed all36 Dense summaries from raw records, recomputed the72-row CSV
and paired intervals, reran37 phase audits and the delivery verifier, and checked
all447 reference outputs. It also reconstructed MoE service/recorder-control
results and checked the overhead, storage and representative rank summaries.
One nonblocking completion-flag weakness was fixed in `e67d563` and independently
re-reviewed: wrong grids/durations now remain incomplete, while the actual
36-point output is unchanged. No blocking findings remain. This is Codex-only
provisional review, not dual-review acceptance.

## Meeting requirement disposition

| Requirement | Delivered result and boundary |
| --- | --- |
| Original InferenceX/AgentX route | Pinned harness integration, original default-route Weka run, scheduler/DAG/snapshot/salt acceptance and nonofficial token-mode provenance. |
| Low-overhead per-token replacement | Normal forward/sampling retained; packed device token storage, metadata-only batch maintenance and replacement before next forward. Scoped matched-output HTTP target passes; no universal overhead guarantee. |
| KV behavior | Full same-content paired matrix with source-conditioned cache counts, sampled residency, evictions and retractions. Roomy cache reuse improves; pressure c8 does not establish improvement. No claim of bitwise KV equality or reduced GPU FLOPs. |
| Warmup concern | All37 audited runs drain warmup HTTP requests before profiling. Warmup compute and cache-state effects are measured separately; no removal or concurrent GPU-interference conclusion is claimed. |
| Public hash-only traces | Original true replies cannot be recovered. Public reference stays separate; target-tokenized Exgentic content and explicit synthetic controls carry the content tests. |
| MoE and multiple GPUs | Full308-request real/synthetic service and both-rank routing coverage, fixed TP2/EP2 placement and EPLB off, representative9-request rank timing and paired recorder-cost controls. Whole-workload rank timing is not established. |
| Reproducibility and delivery | Seed42, source/model hashes, function-scoped commits, two dependency patches, Git LFS logs, complete tables, commands and a local bilingual PR draft. No upstream publication or maintainer acceptance is implied. |
| Review | Deterministic delivery checks and final independent Codex review/re-review complete without blocking findings. Codex review is provisional because Claude-review is unavailable; this is not dual-review acceptance. Shen Yue is not a default review dependency. |

The measurements support reproducible trajectory replay as a benchmark-fidelity
improvement. They do not establish general engine acceleration, a full removal
of warmup, or a causal MoE routing-to-GPU-time mechanism.

The deliverable is an opt-in replay implementation with a bounded benchmark
study and a local PR draft. GPU0/2/3 are released, all experiment stages completed,
and raw evidence is retained in Git LFS with verified archive hashes. A maintainer
can review the interfaces and evidence without treating these local runs as an
official InferenceX score or upstream acceptance.

## Lifecycle follow-up, 2026-11-07

An isolated SGLang copy on gpu6 added opt-in lifecycle observation for selected
request IDs. A seed-42, eight-request qualification from the frozen Dense corpus
passed all 447 source/reference checks. The first server startup stalled during
CUDA graph capture and never became healthy; its log is retained. A correctness
run with CUDA graph disabled became healthy and executed the frozen c4 window and
its single permitted c8 escalation. This configuration change has no performance
interpretation.

Both windows completed without driver errors. Mixed replay/ordinary filtering
and merging (L2) passed, as did active cancellation, backend abort, batch
retirement and post-cancel ordinary/replay followups (L3). The same longest
request never had a reset-and-resume chain, so L1 remains **UNCOVERED**. Test
retraction triggers and resets of other requests do not satisfy that check.
The original receipts, corrected read-only validation, wire records, observer
events, server logs and both frozen specs are in
[`lifecycle-20261107-raw.tar.gz`](../../experimental/token_replay/evidence/lifecycle-20261107-raw.tar.gz).
The local archive SHA-256 is
`0204815e7ff0a624115316f4bb38cfb4237c9444a1444c8b32a539059e0d481f`,
matching gpu6. GPU0 was released after the study. This follow-up does not upgrade
the Codex-only review to cross-family acceptance.

A separate controlled L1 check then used SGLang's documented
`/pause_generation` `mode="retract"` and `/continue_generation` path. The
non-speculative scheduler retracts the shortest current output first, explaining
why the earlier driver did not select its longest request despite many test
triggers. In an isolated no-CUDA-graph service, a selected 641-token request was
paused after eight observed tokens. Its own `reset_for_retract`, resumed extend,
decode, and length finish events formed a complete chain; all 641 wire output
IDs exactly matched the frozen response. The first two attempts and the unique-ID
passing attempt are retained in
[`lifecycle-pause-20261107-raw.tar.gz`](../../experimental/token_replay/evidence/lifecycle-pause-20261107-raw.tar.gz).
This establishes **controlled L1 correctness**, while natural KV-pressure L1
coverage and CUDA-graph startup cause remain unresolved. No performance or
cross-family review claim follows from this check.
