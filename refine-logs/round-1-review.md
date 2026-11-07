# Independent Codex method review

Frozen proposal: `2ce840fb0f5191adba699b6354088ee503950425`. Review status: Codex-only provisional; Claude unavailable.

<details><summary>Raw independent Codex review</summary>

**Verdict: REVISE the planning package, with small specification changes.** The engineering method is appropriate; it does not need a broader research contribution or new architecture. The necessary revisions concern B2’s execution contract and one C1 coverage boundary. B1/B2 need not be completed to approve an actionable plan.

Review basis: frozen commit `2ce840fb0f5191adba699b6354088ee503950425`, its proposal and grounding ledger, the pinned harness and SGLang source, existing lifecycle/overhead scripts, and available evidence summaries. No files were changed and no GPU work was performed.

Reviewer: fresh independent Codex reviewer. **Exact runtime model identifier: unavailable.** This is a **Codex-only provisional** method review. Claude review is missing; earlier delivery reviews do not substitute for this review.

**CALIBRATION: none** — no human-curated anchors were available.

| Axis | Weight | Score | Assessment |
|---|---:|---:|---|
| Problem Fidelity | 15% | 9 | Directly addresses the recorded-response/cache-trajectory mismatch while preserving the requested forward and sampling work. |
| Method Specificity | 25% | 6 | Core integration is concrete. Optional B2 leaves occurrence mapping, exact-ID observation, and finite-session termination insufficiently specified. |
| Contribution Quality | 25% | 8 | Useful opt-in integration across loader, endpoint, serving lifecycle and evidence collection. Engineering differentiation is adequately distinguished from existing token forcing. |
| Frontier Leverage | 15% | 9 | Correctly omits unnecessary learned controllers, kernel determinism, embedding precomputation and scheduler replacement. |
| Feasibility | 10% | 7 | Implemented mechanism and reusable fixtures make the narrow deliverable feasible. B2 remains conditional on an unresolved, but bounded, integration gate. |
| Validation Focus | 5% | 7 | B1 targets the most valuable missing lifecycle event; B2 avoids rerunning the large campaign. Timing and decision details need a short freeze specification. |
| Technical-report readiness | 5% | 8 | Existing results have appropriately limited interpretations. The final report can remain useful even if B2 is deferred or fails matching. |

**Weighted composite: 7.65/10.**

**GAP:** The principal remaining gap is an executable comparison contract for the optional AgentX overhead extension. “Capture normal outputs and replay those outputs” works directly in the existing fixed HTTP experiment, but the inherited AgentX path invokes source rows through snapshot warmup, resumed tails and recycling. Its trace format associates one output sequence with each source row, and its normal OpenAI completion response path does not expose the native `output_ids` field. The planning package must specify how those facts fit its per-invocation matching requirement. This does not invalidate the existing HTTP result, require completed experiments to judge the plan, or justify changing the scheduler.

The core method is the smallest adequate mechanism under the stated requirement. Post-sampling replacement directly distinguishes it from pre-sampling logit forcing. Packed storage and metadata filtering/merging are reasonable implementation choices, and the proposal correctly declines to claim a speed advantage over CustomLogitProcessor without a comparison. The scope remains an integration contribution, with useful bounded measurements.

Concrete revisions needed:

1. **Specify B2’s immutable reference and occurrence mapping.** The loader attaches `output_ids` by source indices; the endpoint sends that sequence or its one-token prefix. It does not currently support a different recorded response for each recycled occurrence. See [weka_trace.py](/Users/marsdon.mac/infra/agentx-token-replay/src/aiperf/dataset/loader/weka_trace.py:100) and [openai_completions.py](/Users/marsdon.mac/infra/agentx-token-replay/src/aiperf/endpoints/openai_completions.py:61).

   The smallest plan is to require a single consistent frozen output sequence per source row, with every warmup occurrence matching the appropriate prefix. Preserve complete input IDs, lengths, timestamps, source identities and scheduling metadata when constructing this overhead reference. Freeze the capture once; if occurrences or subsequent generated runs disagree, stop B2 and retain the mismatch. Do not repair the reference by repeatedly capturing outputs until it passes, substituting source rows, or changing later prompts to incorporate newly generated responses. If the intended method instead needs occurrence-specific responses, describe that added interface explicitly before calling B2 execution-ready.

2. **Resolve exact-ID observation at the measurement boundary.** Existing original-entry verification reads engine `request.finished` events containing `output_ids`; the HTTP overhead test instead receives them directly from `/generate`. The original AgentX endpoint uses `/v1/completions`, whose response models provide text and optional logprobs, rather than native output IDs. See [check_entry_log.py](/Users/marsdon.mac/infra/DeterminInference-main/experimental/token_replay/check_entry_log.py:58), [measure_overhead.py](/Users/marsdon.mac/infra/DeterminInference-main/experimental/token_replay/measure_overhead.py:59), and [protocol.py](/Users/marsdon.mac/infra/sglang/python/sglang/srt/entrypoints/openai/protocol.py:353).

   State how every timed invocation will be checked while detailed diagnostics are disabled. An instrumented comparison is permissible if instrumentation is identical and the claim explicitly names that boundary. A separate diagnostic run must not be described as exact-ID verification of every unobserved timing run. The simpler alternative is to defer B2 and retain C2’s already-supported HTTP scope.

3. **Turn the finite-session CPU gate into explicit pass/fail behavior.** Define whether “four-session cohort” means four loaded source sessions, four profiling root admissions, or both. The wrapper currently always forwards duration ([replay.py](/Users/marsdon.mac/infra/DeterminInference-main/inferencex-e2e/infx/bench/agentic/replay.py:158)); session and duration stop conditions have different effects ([stop_conditions.py](/Users/marsdon.mac/infra/agentx-token-replay/src/aiperf/timing/phase/stop_conditions.py:152)).

   Before GPU work, require concrete CPU fixtures showing that a root resumed after snapshot handoff counts exactly once, its remaining turns drain, no replacement root is admitted after the cap, and the process terminates without cancellations or pending work. Include a gated-parent/child fixture only if that path is included in the claimed finite-cohort support. Declare that a duration/watchdog stop before cohort completion makes the run incomplete. Current source already includes explicit resumed-root accounting, so this is a validation requirement, not evidence that a new scheduler is needed.

4. **Close the remaining acceptance and budget wording.** B1 should bind an observed retraction and resume to the same replay request and demonstrate output continuity after resumption. Freeze its pressure setting and single escalation before execution. The contract mentions cancellation cleanup, but B1 and the cited mixed-request fixture do not explicitly exercise cancellation. Either add one focused cancellation-to-ordinary-request witness or label cancellation untested in C1’s supported-path boundary.

   For B2, state that the 1.5 GPU-hour ceiling includes capture, calibration, startup, warmup, measurement and drain. Specify whether calibration is excluded from inference, reuse the existing adjusted 3/5-pair interval rule explicitly, and stop without a positive C2 extension if the remaining budget cannot complete the required look. Do not shrink or replace the cohort after seeing results.

These changes address the only sub-7 dimension, Method Specificity. They can be made in the plan without launching B1/B2.

C1 is appropriately limited to declared output IDs, salt isolation and explicitly tested lifecycle paths. The existing exact-output and cached-followup witnesses should remain credited; a universal KV tensor audit is unnecessary. C2 is justified only at the declared fixed-prompt HTTP boundary by E4. The Dense closed-loop matrix measures a different workload comparison and cannot supply pure replacement overhead. The proposal recognizes this correctly.

The best simplification is to make **B1 the sole planned pre-merge evidence addition** and defer B2 unless a wider AgentX overhead claim is required. No new MoE study, scheduler, embedding path, controller, deterministic kernel suite or broader model campaign is warranted. Modernization is unnecessary for this pinned engineering delivery.

**Drift warning:** session-cap forwarding or lifecycle fixes would create a new code freeze requiring focused re-review. Do not carry acceptance across changed harness/serving versions, extend c1 evidence to c8, convert source-row counts into profiling-call coverage, or reinterpret matched returned IDs/cache counters as identical internal computation. GPU assignment and scheduling remain unchanged; this review recommends no GPU launch or resource expansion.

<oai-mem-citation>
<citation_entries>
MEMORY.md:228-237|note=[prior scope and matching constraints checked against current source]
</citation_entries>
<rollout_ids>
01a1ac90-e4e8-7f23-96a9-37e932899f8f
</rollout_ids>
</oai-mem-citation>

</details>
