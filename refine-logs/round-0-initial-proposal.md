# Engineering Proposal: Recorded-token replay for AgentX

Target: reviewable engineering PR and bounded technical report. No target academic venue is assumed. This proposal refines an implemented extension; it does not relabel completed results as planned or pre-registered.

## Problem Anchor

- Bottom-line problem: Provide an opt-in, reproducible recorded-token path for the existing InferenceX AgentX benchmark, so later recorded prompts can reuse cache corresponding to recorded response tokens while the engine still performs normal forward and sampling.
- Must-solve bottleneck: Response-only replay can return or insert recorded text while the server has computed a different token trajectory; the extension must carry exact token IDs through the supported scheduler, cache-isolation and request lifecycle without silently changing the default benchmark.
- Non-goals: A new determinism algorithm, bitwise equality of floating-point KV tensors across batches, universal speedup, removal of snapshot warmup, reconstruction of original content from hashes, model training, or a top-venue novelty claim.
- Constraints: Engineering PR and technical report first. Reuse this repository's qualified Exgentic/Weka sources, existing Dense/MoE checkpoints and pinned AgentX 1.0.13/SGLang 0.5.10. Seed42; function-scoped code commits; raw execution evidence in Git LFS. Future GPU work requires a fresh resource guard and isolated outputs; GPU1 is outside scope. This pipeline plans follow-up work and does not launch it.
- Success condition: A maintainer can inspect the opt-in API, inherited versus added behavior, exact-ID/lifecycle coverage, declared overhead boundary, default-route regression evidence, reproducible artifacts and honest limitations; each remaining claim has a small falsifiable test and a stopping rule.

## Method thesis and contribution

**Thesis:** Add an explicit recorded-token mode to the existing AgentX path that carries paired prompt/response IDs and replaces each normally sampled next ID before its next model forward, preserving the supported scheduler and cache-isolation contract with separately measured mechanism cost.

The dominant contribution is a small serving/benchmark integration with reproducible regression evidence. A supporting contribution is the bounded technical report on cache reuse, phase costs and content-dependent MoE routing. Token forcing, DAG scheduling, warmup and batch-invariant determinism are not new contributions. No trainable component, learned policy or frontier-model controller is required.

## Technical gap and route choice

Response-only replacement cannot change a server's already-computed token path. Shape/hash-only traces cannot supply original response IDs. Reusing normal sampling then replacing the chosen ID is the smallest intervention that satisfies the meeting contract.

Route A, selected: the implemented post-sampling replacement, packed device IDs and metadata-only filter/merge bookkeeping. Preserve source-model forward, sampler, normal embedding lookup and all inherited scheduling behavior.

Route B, not selected for this contract: force a token with the existing CustomLogitProcessor. It is a valid existing engineering precedent, but changes the distribution before sampling and cannot by itself demonstrate that the original sampler workload was preserved. Do not claim that Route A is faster without a separate valid comparison. No new performance arm is required to explain this interface distinction.

Adding batch-invariant kernels, precomputed embeddings, a learned controller, a new trace scheduler or an expert balancer would broaden scope without addressing an observed extension-specific bottleneck. Full-KV equality and a warmup redesign are explicitly excluded.

## Concrete system contract

1. Loader accepts paired input_ids/output_ids from supported token-bearing traces; preserve source row, session/reset and DAG metadata. Reject mismatched lengths/budgets and ambiguous template reconstruction. Public hash-only data stays on its original path.
2. InferenceX explicitly selects AGENTX_TOKEN_TRACE and AGENTX_REPLAY_MODE=generated|recorded with the pinned wheel. Token-mode results remain submission_valid=false. Unset token mode retains default behavior.
3. Existing AgentX scheduler selects snapshot/warmup and profiling work. Complete prompt IDs bypass chat templating; one-token warmup truncates replay IDs to one. Tree members share a cache salt; independent trees are isolated without modifying prompt IDs.
4. SGLang stages response IDs in device storage. Each decoding step performs normal model forward and sampling, then substitutes the recorded token before subsequent forward and logprob reporting. Indexing must follow the request's consumed output prefix across filter/merge/retraction/resume; no out-of-range or stale state after completion/cancellation.
5. Collect exact-output/lifecycle evidence with diagnostics enabled. Performance measurements disable detailed diagnostics and retain normal request/phase metrics. The last output token's actual forward behavior is implementation-specific and must be observed when reasoning about internal work.

Training/loss: not applicable. Model and weights remain frozen. Optimizations already implemented are packed IDs and metadata maintenance; normal embedding lookup stays intact because existing measurements do not identify it as an incremental bottleneck.

## Claims and acceptance boundaries

**C1 — supported-path replay correctness:** Admitted recorded-token requests preserve their declared output IDs, salt isolation and ordinary-request behavior across the explicitly tested lifecycle paths. Existing support is E1/E3; a targeted retraction/resume exact-ID regression remains the highest-value missing coverage. C1 is not a theorem of bitwise or global-residency equivalence.

**C2 — scoped incremental cost:** In a declared matched-output and matched-cache-work experiment, replay adds no more than3% end-to-end time at the stated boundary and conditions. E4 already supports the fixed32-prompt HTTP claim. A claim at the original AgentX scheduler boundary requires a separate matched finite-cohort measurement; E2 cannot establish pure replacement overhead because work/mix change.

Anti-claims: correct returned IDs alone prove all cache state; reduced cache misses imply proportional throughput gains; warmup overlaps profiling in this pinned version; MoE routing differences imply throughput changes; full447-row coverage follows automatically from a finite benchmark cap. All are rejected.

## Minimal validation sketch

**B0 — reuse/freeze, already complete:** Attach the implemented two dependency patches, existing exact-ID/DAG/salt/cached-followup witnesses,36-point study,447 reference, HTTP cost and MoE study. Preserve failures and limitations. No repeat of the9-hour matrix and no new MoE model runs. The generated/recorded comparison is the mechanism-deletion control already available; it is not a pure overhead control.

**B1 — targeted lifecycle regression, recommended pre-merge gate:** Reuse the existing mixed/lifecycle fixture and selected qualified requests to observe one actual retraction followed by resume, exact returned IDs, filter/merge with unequal lengths and normal requests after replay state exits. Require the observed event, not just a configured pressure label or a counter from an unrelated performance run. Bind expected IDs, lengths, salts and source provenance before execution. Use one baseline pressure condition and at most one predeclared escalation; preserve noncoverage/failures. Cached-versus-fresh/logprob or cutpoint KV comparisons are optional diagnostics if a mismatch appears, not a universal KV acceptance requirement.

**B2 — original-path cost extension, conditional:** Only needed to expand C2 beyond HTTP. First verify minimal forwarding of the existing session cap and stop/drain behavior on concrete CPU fixtures. Use a fixed four-session Dense cohort with seed42 and unchanged snapshot/warmup, initially c1. Freeze the actual expanded warmup/profiling invocation ledger; do not promise447 profiling requests or whole original sessions. Capture normal greedy outputs, replay those same outputs, verify per-invocation IDs, lengths, cache work, salt equivalence groups, zero retractions and no cancellations. Three alternating paired repetitions; expand directly to five only if the prespecified interval crosses3%. Any workload mismatch invalidates isolation and is retained, not filtered or recaptured until it passes. A high-concurrency claim requires its own frozen c8 cohort; it is not inferred from c1.

## Planning gate

The mechanism is stable and implemented. Dominant contribution: explicit recorded-token integration plus narrow regression/measurement contracts. Rejected complexity: new training, embedding precomputation, deterministic kernels, new scheduler, full KV tensor audit, repeated Dense/MoE campaigns. Frontier primitive: absent as a new method component; an LLM is the workload under test. Remaining issues: bounded retraction coverage, valid finite-cohort stopping/matching if C2 is expanded, and missing Claude review. They require implementation/measurement or reviewer availability, not a broader story.

## Cost and stop rules

Planning performs no GPU work. B0 is CPU/read-only reuse. Reserve up to0.5 GPU-hours on GPU0 for B1 after a fresh guard, with two bounded attempts maximum; failure/noncoverage stops expansion. B2 first has a CPU integration gate and then one calibration pair. Its measured wall times must determine the final3/5-pair budget; initial planning envelope is at most1.5 GPU-hours for c1, excluding any separately approved scope expansion to c8. Existing HTTP32-prompt c1 arm is about40s; this is only a lower-scope reference, not an AgentX runtime estimate. No new datasets/models or other-project data.

## Risks and proposed decision

The current bounded technical report and PR draft remain usable. Treat B1 as the priority acceptance improvement before merge; do B2 only if the PR claims the wider AgentX boundary. If B1 exposes a bug, fix only the owning lifecycle function and re-review/retest that path. If B2 cannot match work, retain the already supported HTTP wording rather than invent a scheduler or claim a universal3% result. Maintainer approval and official benchmark acceptance are outside this planning artifact.

Sources and existing run identities: [GROUNDING.md](GROUNDING.md). This candidate has not yet received the new independent method review. Earlier delivery review is not substituted for that review.
