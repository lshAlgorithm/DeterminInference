# Refinement Report

Date: 2026-11-07. Target: engineering PR and technical report. Final planning verdict: **READY**, **9.15/10**, **provisional**. Three independent Codex rounds; Claude review missing. This workflow changed planning documents only, launched zero GPU jobs and did not publish or merge a PR.

## Problem Anchor

- Bottom-line problem: Provide an opt-in, reproducible recorded-token path for the existing InferenceX AgentX benchmark, so later recorded prompts can reuse cache corresponding to recorded response tokens while the engine still performs normal forward and sampling.
- Must-solve bottleneck: Response-only replay can return or insert recorded text while the server has computed a different token trajectory; the extension must carry exact token IDs through the supported scheduler, cache-isolation and request lifecycle without silently changing the default benchmark.
- Non-goals: A new determinism algorithm, bitwise equality of floating-point KV tensors across batches, universal speedup, removal of snapshot warmup, reconstruction of original content from hashes, model training, or a top-venue novelty claim.
- Constraints: Engineering PR and technical report first. Reuse this repository's qualified Exgentic/Weka sources, existing Dense/MoE checkpoints and pinned AgentX 1.0.13/SGLang 0.5.10. Seed42; function-scoped code commits; raw execution evidence in Git LFS. Future GPU work requires a fresh resource guard and isolated outputs; GPU1 is outside scope. This pipeline plans follow-up work and does not launch it.
- Success condition: A maintainer can inspect the opt-in API, inherited versus added behavior, exact-ID/lifecycle coverage, declared overhead boundary, default-route regression evidence, reproducible artifacts and honest limitations; each remaining claim has a small falsifiable test and a stopping rule.

## Focused result

The core method remains the delivered opt-in AgentX integration: carry complete recorded token IDs, run ordinary forward and sampling, and substitute the recorded next ID before subsequent forward. Packed storage and metadata maintenance support the mechanism. Existing scheduler, snapshots, warmup, cache salts and normal embedding lookup remain the foundation. Token forcing itself is not claimed as novel.

The final claims distinguish supported-path correctness from scoped incremental cost. Existing full-study evidence is retained as B0; it is not retrospectively described as a pre-registered consequence of this plan. The 240-request entry witness and 447-request reference remain credited. HTTP overhead supports the fixed-prompt boundary; Dense closed-loop cache effects cannot isolate replacement cost. MoE routing differences do not establish throughput effects. Warmup HTTP drain does not make warmup free or establish GPU-event non-overlap.

## Refinement decisions

1. Make B1 the sole recommended pre-merge evidence addition: actual same-request retraction/resume, unequal-length batch transitions and explicit cancellation cleanup. Use the existing test trigger on the production retraction function, safe capacity, c4 and one c8 escalation; record triggered-path scope.
2. Require a small opt-in observer and dedicated driver before running B1. Existing counters do not prove the event chain. These functions are TODO; planned commands are not presented as executable artifacts.
3. Correct the streaming-abort validator specification after independent source inspection: nonterminal packets and abort terminal receipts have different contracts. This fixes a potential false test failure, not an observed serving-code defect.
4. Defer B2 unless a broader overhead claim is required. Its CPU gate proves finite-session accounting/drain; one immutable reference and complete symmetric output-ID instrumentation define an instrumented AgentX boundary. Failure to match stops extension without invalidating the existing HTTP result.
5. Preserve hard, inclusive budgets: <=0.5 GPU-hours for B1, conditional <=1.5 GPU-hours for B2. Seed 42, no new data/model, raw evidence in Git LFS, function-scoped code commits and focused re-review on future changes.

Rejected complexity: scheduler replacement, embedding precompute without measured need, deterministic kernel suite, learned controller, expert-balancing feature, universal KV theorem and repeated Dense/MoE campaigns. The LLM is the workload, not a new research primitive.

## Review evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Technical-report readiness | Composite | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 9 | 6 | 8 | 9 | 7 | 7 | 8 | 7.65 | REVISE |
| 2 | 9 | 8 | 9 | 10 | 9 | 9 | 9 | 8.90 | REVISE |
| 3 | 9 | 9 | 9 | 10 | 9 | 9 | 9 | 9.15 | READY |

The initial review sharpened interfaces and scope; the second found the abort packet exception; the third inspected the correction. Raw reviews and per-round full proposal records remain preserved. [REVIEW_SUMMARY.md](REVIEW_SUMMARY.md) distinguishes each freeze and scope. CALIBRATION:none; all positive Codex judgments remain provisional. [REVIEW_AVAILABILITY.md](REVIEW_AVAILABILITY.md) records the unavailable Claude counterpart.

## Deliverables and next action

- [FINAL_PROPOSAL.md](FINAL_PROPOSAL.md): clean method and C1/C2 boundaries.
- [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md): data, controls, decisive metrics, implementation prerequisites, budgets and stopping rules.
- [EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md): completed B0, TODO B1 and deferred B2.
- [PIPELINE_SUMMARY.md](PIPELINE_SUMMARY.md): compact handoff.

Next is B1 CPU implementation and protocol freeze, followed by source review and a separately recorded bounded execution. No pending lifecycle or cost result is marked complete by this planning workflow.
