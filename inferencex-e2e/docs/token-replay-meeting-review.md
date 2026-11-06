# Token replay meeting implementation and evidence

The original `agentx profile` route now accepts complete recorded token IDs,
retains the AgentX scheduler, snapshot warmup, spawn/join and session isolation,
and runs through the InferenceX result collector. The full performance study is
still running. Neither the 3% overhead target nor an improvement in benchmark
accuracy is established by the acceptance runs below.

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

| KV capacity | Profiling requests | Cache hit fraction from server aggregate | Peak KV occupancy | Evicted tokens including warmup |
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

## MoE experiment protocol

The BF16 Qwen1.5-MoE-A2.7B-Chat weights are fixed at revision
`ec052fda178e241c7c443468d2fa1db6618996be`. GPU2/3 use TP2/EP2, trivial fixed expert
placement, EPLB disabled, and separate shared experts. The isolated environment
and a fresh agent following the documented invocation passed the run6 witness.
That witness proves feasibility and exact 32-token replay, not workload effects.

The full service comparison runs all 308 qualified requests in both real and
synthetic arms, three paired repetitions with alternating order. Recorder is
absent from the performance engine. Output IDs are checked exactly on every
request, with request latency, client TTFT and cached-token counts retained.
Service throughput divides output count by summed request wall time; it excludes
flush, file writes and between-session gaps and is not whole-campaign throughput.

Recorder-on runs and per-rank compute/communication diagnostics are separate.
Recorder histograms are global and replicated across ranks: accumulate sessions
within one rank, never sum the two replicas. Token load imbalance is not GPU
execution-time imbalance. The initial first-request profile covers only 1/308
requests and cannot stand in for workload-wide rank timing.

## Review and remaining completion gates

Fresh Codex reviews found and prompted fixes for target-template argument
encoding, full-trace admission, DAG-check false positives, HTTP budget-check
false positives, cache-flush response parsing and overhead warmup/stopping rules.
The salt-forwarding bug was established by a real failed isolation witness.
Deterministic/live checks support the specific claims above. Codex positive
reviews remain provisional. No `claude-review` MCP tool was available, so no
cross-family review or dual-review acceptance is claimed.

The original-path implementation/short acceptance and capacity qualification are
complete. The full Dense matrix, final overhead decision, public-Weka GPU run,
MoE recorder/control perturbation and representative per-rank timing, final
result interpretation, and final PR validation remain completion gates. Running
jobs and partial tables do not satisfy those gates.
