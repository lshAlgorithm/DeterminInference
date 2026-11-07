# Refinement Report

The target is an engineering PR and technical report. The core implemented mechanism is retained: transport complete recorded IDs through the original AgentX route, preserve forward and sampling, and substitute the next token before subsequent forward. This workflow refines claims and follow-up verification; it launches no GPU jobs and changes no serving code.

## Decisions

- Keep one dominant integration contribution and at most the existing bounded systems report as support.
- Preserve the Problem Anchor verbatim. Do not promote token forcing, inherited scheduling or warmup into novelty.
- Reuse the complete Dense/MoE delivery as B0. Preserve observed negative/inconclusive results and source/measurement limits.
- Make B1 the sole recommended pre-merge addition: observed request-level retraction/resume continuity and cancellation cleanup.
- Defer B2 unless a broader AgentX overhead claim is necessary. It requires a finite-session CPU gate, immutable source-row capture, exact observation in every timed run, symmetric instrumentation and an all-inclusive budget.
- Reject a new scheduler, embedding precompute, deterministic kernel suite, expert-balancing feature and new benchmark campaign.

## Evidence and review boundaries

Existing HTTP overhead supports a scoped 3% result. Closed-loop Dense generated-versus-recorded effects cannot supply pure replacement overhead. Existing exact-ID and cached-followup witnesses remain credited, but do not imply universal KV or lifecycle equivalence. Warmup HTTP drain is established; zero warmup cost and GPU-event non-overlap are not. MoE content changes routing, while throughput separation is not established.

The initial independent review scored 7.65/10 and requested specification changes. Responses are in [round-1-refinement.md](round-1-refinement.md). Claude review is unavailable, so positive Codex judgments remain provisional. Full history and latest verdict are maintained in [REVIEW_SUMMARY.md](REVIEW_SUMMARY.md).

## Handoff

[FINAL_PROPOSAL.md](FINAL_PROPOSAL.md) freezes C1/C2. [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) and [EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md) distinguish completed evidence, pending lifecycle work and conditional scope expansion. No result in the follow-up plan is represented as already executed.
