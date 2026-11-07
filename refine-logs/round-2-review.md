<details><summary>Raw independent Codex review</summary>

**Verdict: REVISE — one narrow B1 protocol correction.** The package is a focused, actionable engineering plan. It does not need additional experiments, a broader contribution, or a new architecture. B1/B2 need not have run for the plan to be reviewable.

Reviewed freeze: `533a811249abc7e21ae69e3b20769a741d65bb5a`. I read all seven requested planning artifacts and inspected the relevant pinned harness/SGLang paths. I formed the source-based finding before consulting the earlier review and refinement responses.

Reviewer: fresh independent Codex reviewer. **Exact runtime model identifier: unavailable.** Positive judgments are **Codex-only provisional**. Claude review is unavailable; this is not dual-review acceptance.

**CALIBRATION: none** — no curated anchors.

| Axis | Weight | Score | Assessment |
|---|---:|---:|---|
| Problem Fidelity | 15% | 9 | Preserves the recorded-token/cache-trajectory problem and the requirement to retain normal forward and sampling. |
| Method Specificity | 25% | 8 | Detailed lifecycle and optional timing contracts; one streaming-abort assertion contradicts the pinned implementation. |
| Contribution Quality | 25% | 9 | Useful opt-in integration with reproducible regression boundaries, appropriate for an engineering PR/report. |
| Frontier Leverage | 15% | 10 | Correctly omits unnecessary frontier components, scheduler replacement, and new model campaigns. |
| Feasibility | 10% | 9 | Reuses implemented mechanisms and narrow fixtures, with explicit preparation gates and hard budgets. |
| Validation Focus | 5% | 9 | B1 targets missing lifecycle coverage; B2 remains a conditional claim expansion. |
| Technical-report readiness | 5% | 9 | Claims distinguish cache consequences, HTTP cost, instrumented AgentX cost, and limited MoE observations. |

**Weighted total: 8.90/10.**

**GAP:** The remaining planning gap is an incorrect universal assumption about streaming output packets during cancellation. Normal streaming packets are cumulative with the default configuration, but an abort delivered while a progressed request is back in the waiting queue takes a terminal path that emits only its last output token. Applying the plan’s cumulative-prefix assertion to that packet can classify correct backend cancellation as replay corruption. This is a small driver-specification issue, not evidence of a replay implementation defect, a reason to repeat B0, or a requirement to execute B1 before approving the plan.

**Required correction: distinguish normal streaming packets from terminal abort receipts.**

[EXPERIMENT_PLAN.md:55](/Users/marsdon.mac/infra/DeterminInference-main/refine-logs/EXPERIMENT_PLAN.md:55) states that streaming responses contain cumulative output arrays and directs validation of each as an expected prefix. The pinned implementation has two relevant qualifications:

- Ordinary packets are cumulative only when `incremental_streaming_output=false`; see [tokenizer_manager.py:1620](/Users/marsdon.mac/infra/sglang/python/sglang/srt/managers/tokenizer_manager.py:1620).
- `_handle_abort_req` sets `completion_tokens` to the complete retained output count but, for streaming requests, emits `output_ids=[last_output_id]`; see [tokenizer_manager.py:2188](/Users/marsdon.mac/infra/sglang/python/sglang/srt/managers/tokenizer_manager.py:2188). A progressed replay request can enter the waiting queue through retraction and then be aborted through [scheduler.py:3180](/Users/marsdon.mac/infra/sglang/python/sglang/srt/managers/scheduler.py:3180).

For a consumed prefix `[a,b,c]`, the terminal packet may therefore contain `[c]`, which is not the prefix `[a]`. This is permitted terminal formatting.

The minimal fix is to pin `incremental_streaming_output=false`, apply cumulative-prefix checks to nonterminal normal packets, and validate terminal abort packets separately using the same RID, `finish_reason.type=abort`, `completion_tokens`, retained prefix, and backend event chain. A singleton terminal array must not be concatenated as a new token or interpreted as a replay restart. Preserve the existing follow-up ordinary/fresh-replay checks and state-retirement requirement. No new experiment block is needed.

I verified that both the scheduler and tokenizer-manager files in `serving-source-residency-20261106.tar.gz` are byte-identical to the inspected sibling sources. The tokenizer-manager SHA256 is `ddabdab752b3d8dd2be3cc8c033f5eab1c2c76e135ab57498643203bfc18079b`.

The remaining scrutinized contracts are sound at the planning level:

- **Retraction/resume:** The test predicate and KV-exhaustion predicate reach the same `batch.retract_decode()` call at [scheduler.py:2574](/Users/marsdon.mac/infra/sglang/python/sglang/srt/managers/scheduler.py:2574). The huge no-prefill threshold permits re-admission. `retract_decode()` retains at least one running request, so the mixed c4/c8 fixture matters. The plan correctly requires actual same-RID retraction, resumed input construction, subsequent progress, and complete exact outputs; a global counter cannot pass L1.
- **Cancellation:** Requiring the original request’s abort terminal response is correct. Running-request abort can permit a cleanup decode, as documented at [scheduler.py:3266](/Users/marsdon.mac/infra/sglang/python/sglang/srt/managers/scheduler.py:3266). Unique non-prefix RIDs correctly account for prefix matching.
- **Finite cohort:** Four loaded sessions and four profiling root admissions are distinct, explicit declarations. The CPU gate requires once-only accounting for snapshot-resumed roots, tail drain, no fifth admission, and no outstanding/error/cancelled work. Existing [credit_counter.py:261](/Users/marsdon.mac/infra/agentx-token-replay/src/aiperf/timing/phase/credit_counter.py:261) and [stop_conditions.py:152](/Users/marsdon.mac/infra/agentx-token-replay/src/aiperf/timing/phase/stop_conditions.py:152) support this design. Wrapper forwarding remains future work and requires a new freeze.
- **Immutable reference:** One output vector per source row, repeated-occurrence agreement, warmup-prefix agreement, and refusal to recapture until matching prevent reference repair after seeing results. The warmup-only-row gate may defer B2; that is an honest outcome for optional scope, not a promise of successful capture.
- **Observation boundary:** Symmetric `request.finished` output-ID instrumentation, complete occurrence mapping, and inclusion of required observation drain in timing justify the explicitly instrumented boundary. Separate diagnostics are correctly excluded as certification of unobserved timing runs.
- **Stopping rule:** The n=3/n=5 constants and inclusive 3% containment rule match [measure_overhead.py:71](/Users/marsdon.mac/infra/DeterminInference-main/experimental/token_replay/measure_overhead.py:71). Calibration is excluded, intermediate looks are forbidden, and fixed-cohort execution variability is distinguished from population uncertainty.
- **Budgets:** B1’s 0.5 GPU-hour ceiling includes initialization, references, attempts, drain, and cleanup. B2’s separate 1.5 GPU-hour ceiling includes capture and calibration as well as measurements. Budget exhaustion cannot produce a positive extension.

One nonblocking editorial correction would improve status consistency: [FINAL_PROPOSAL.md:79](/Users/marsdon.mac/infra/DeterminInference-main/refine-logs/FINAL_PROPOSAL.md:79) uses `INCONCLUSIVE` when the remaining budget cannot finish the required look, while [EXPERIMENT_PLAN.md:120](/Users/marsdon.mac/infra/DeterminInference-main/refine-logs/EXPERIMENT_PLAN.md:120) uses `INCOMPLETE` for runtime exhaustion. Reserve `INCOMPLETE` for unfinished execution and `INCONCLUSIVE` for a completed five-pair interval crossing the target; insufficient predicted capacity before measurement can remain `DEFERRED`.

No other concrete planning blocker emerged from this review. Correct the terminal-packet specification and perform a focused re-review of the new freeze. Keep B0 credited, B1 pending, and B2 deferred.

**Independence and execution statement:** I changed no files, created no commits, launched no GPU jobs, and did not run lifecycle experiments. A final task-specific diff confirmed that the seven reviewed artifacts still matched `533a811`. Source inspection establishes interface facts; it does not establish empirical lifecycle correctness or merge readiness.

<oai-mem-citation>
<citation_entries>
MEMORY.md:228-237|note=[prior scope and matching constraints checked against current source]
</citation_entries>
<rollout_ids>
01a1ac90-e4e8-7f23-96a9-37e932899f8f
</rollout_ids>
</oai-mem-citation>

</details>
