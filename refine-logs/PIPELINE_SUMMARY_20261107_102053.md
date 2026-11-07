# Pipeline Summary

**Problem:** Recorded-prompt replay needs the server to follow recorded response IDs while retaining normal forward and sampling.

**Final method thesis:** Carry paired token IDs through the existing AgentX path and replace sampled next IDs before subsequent forward, with explicit lifecycle correctness and separately scoped cost evidence.

**Target:** Engineering PR and technical report.

**Final planning verdict:** READY, 9.15/10, Codex-only provisional. Claude unavailable; dual-review acceptance is not established.

**Date:** 2026-11-07. **GPU jobs launched by this workflow:** 0.

## Final deliverables

- [Proposal](FINAL_PROPOSAL.md)
- [Review summary](REVIEW_SUMMARY.md) and [full refinement report](REFINEMENT_REPORT.md)
- [Experiment plan](EXPERIMENT_PLAN.md) and [tracker](EXPERIMENT_TRACKER.md)
- [Grounding and existing evidence](GROUNDING.md)

## Contribution snapshot

Dominant contribution: opt-in serving/benchmark integration, packed replay storage and reproducible supported-path regression evidence. Supporting report: existing cache, phase-cost and MoE observations. Rejected additions: new scheduler, precomputed embeddings without a measured bottleneck, learned controller, new deterministic kernels, expert balancer or repeated model campaign. No new frontier component is needed.

## Claims

**C1 — supported-path replay correctness:** Admitted recorded-token requests preserve their declared output IDs, salt isolation and ordinary-request behavior across the explicitly tested lifecycle paths. This does not assert bitwise KV equality or untested lifecycle coverage.

**C2 — scoped incremental cost:** Under a declared matched-output and matched-cache-work protocol, replay adds at most 3% end-to-end time at the stated measurement boundary and conditions. Current support is limited to the fixed 32-prompt greedy HTTP experiment; a full AgentX overhead guarantee is not established.

## Present evidence and limits

B0 is already complete: 240 original-entry exact-ID requests, 447 real-content reference requests, the full 36-point Dense matrix, HTTP cost and MoE study. Roomy cache reuse changes about +1.65–1.66 percentage points, with modest throughput changes. Pressure c4/c8 and MoE throughput intervals span zero. The HTTP upper adjusted overhead bounds are 0.8150%, 0.3886% and 0.6693% at c1/4/8. Those figures do not establish full-AgentX overhead or universal benefit. Warmup HTTP requests drain before profiling but warmup still has measurable cost. Detailed limits and raw artifact links remain in the existing report.

## First follow-up work

1. **B1 CPU preparation:** implement/review the diagnostic observer and dedicated lifecycle driver; freeze source rows/IDs, event schema, controls and commands. Pin streaming packet semantics, including the abort-terminal exception.
2. **B1 bounded lifecycle execution, future:** safe Dense capacity, deterministic production-path retraction trigger, c4 then at most one c8 escalation. Require same-request retraction/resume/exact output, actual batch transitions and abort cleanup. Total allocation <=0.5 GPU-hours; absent events remain UNCOVERED. Current status: TODO.
3. **B2 only if the claim needs expansion:** first pass finite-session CPU and observation gates; then one frozen reference, calibration and 3/5 paired c1 runs under the all-inclusive <=1.5 GPU-hour cap. Instrumented AgentX scope only. Current status: DEFERRED.

No nine-hour Dense rerun or new MoE study is planned. Follow-up uses seed 42 and repository-supported sources. New code is committed at function scope; raw execution evidence goes to Git LFS. Resource availability must be checked immediately before any later launch.

## Main risks and stopping rules

A retraction counter alone cannot certify resumption; a client disconnect cannot certify backend cancellation. Observe the same RID and real event chain, and distinguish nonterminal cumulative packets from terminal abort receipts. Any exactness error is FAIL; no desired lifecycle event is UNCOVERED.

A fixed source row may produce different outputs across AgentX occurrences, or a finite cohort may not drain as expected. Any mismatch stops B2 with evidence preserved; never repair the reference by recapturing until it passes. INCOMPLETE means unfinished execution, INCONCLUSIVE means a completed five-pair interval still crossing 3%, and DEFERRED means the extension did not start. The existing scoped HTTP result remains intact.

## Review and next action

Three fresh independent Codex reviews produced the focused plan and corrected one streaming-abort assertion. Their positive judgments are provisional. Claude-review was not callable; see [availability receipt](REVIEW_AVAILABILITY.md). Document consistency checks do not replace that missing semantic review.

Proceed with B1 CPU implementation and a new reviewed execution freeze when continuing implementation. The planning package is delivered; the tracker deliberately keeps future empirical work open.
