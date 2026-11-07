# Engineering Grounding and Evidence Ledger

Date: 2026-11-07. Target: engineering PR and technical report. This is a bounded primary-source check, not an exhaustive novelty search. A read-only grounding agent inspected local frozen source first and retrieved official pages/paper metadata over HTTP. No model/data download or GPU work occurred.

## Closest primary sources

| ID | Verified source | Existing capability | Boundary for this extension |
| --- | --- | --- | --- |
| S1 | [AgentX FAQ at60a094f](https://github.com/SemiAnalysisAI/agentx-harness/blob/60a094f256b42f55fde1efb3d45672e1d9a44566/docs/benchmark-modes/semianalysis-agentx-faq.md), SemiAnalysisAI; commit metadata HTTP200, 2026-11-03 | Synthetic content with recorded traffic shape/block prefixes, fixed output lengths, snapshot warmup, salts, DAG and closed-loop scheduling | These are inherited features. Add token-bearing data and serving integration; do not claim to invent AgentX scheduling. |
| S2 | [SGLang sampler at1519acf](https://github.com/sgl-project/sglang/blob/1519acf37c23f2189adb93f57ca9cd2db1bebf18/python/sglang/srt/layers/sampler.py) and [deterministic inference](https://github.com/sgl-project/sglang/blob/1519acf37c23f2189adb93f57ca9cd2db1bebf18/docs/advanced_features/deterministic_inference.md); local Git objects inspected | CustomLogitProcessor modifies logits before sampling; existing tests force selected tokens. Batch-invariant operators are separately supported | Forcing a token is established. The concrete distinction is preserving normal forward AND sampling before replacing the next ID, with packed storage and lifecycle integration. |
| S3 | Qin et al., [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079), official [repository](https://github.com/kvcache-ai/Mooncake/blob/0ff71517f0f80a198eebff3009834ed00504b982/README.md); HTTP200 | Anonymized timing/length/remapped block-hash traces support cache simulation/evaluation | Hash structure does not recover original tokens; do not present a reconstructed reply as original content. |
| S4 | Agrawal et al., [Vidur: A Large-Scale Simulation Framework For LLM Inference](https://arxiv.org/abs/2405.05465v2), [official repository](https://github.com/microsoft/vidur/tree/abae7f63aa857300f5cdc6f5e0d27860cd24721b); HTTP200 | Operator profiling and predictive simulation for deployment/scheduling | This work runs a real engine; it does not claim simulator cost benefits or prediction accuracy. Reading scope: abstract and official repository. |
| S5 | Horace He, [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/), 2025-09-10; HTTP200 | Batch-invariant numerical operations address reduction-order effects | Recorded IDs do not establish bitwise logits/KV/routing equality across batches. |
| S6 | Jiang et al., [Mixtral of Experts](https://arxiv.org/html/2401.04088v1), 2024-01-08; metadata/full relevant text HTTP200 | Content-dependent routing and temporal locality | Content affects expert distributions; this alone does not imply GPU imbalance, communication or throughput effects on Qwen MoE. |

S1/S2 are the decisive engineering comparators. S3-S6 limit the technical report's interpretation; none requires a new component or benchmark. Search absence was not treated as novelty evidence.

## Existing evidence to reuse

| ID | Artifact | Supported statement | Limit |
| --- | --- | --- | --- |
| E1 | [Meeting report](../inferencex-e2e/docs/token-replay-meeting-review.md) | Original entry:240 exact wire/output requests; DAG/reset/salt/mixed lifecycle and budget rejection witnesses | Synthetic acceptance fixture is not real corpus performance. |
| E2 | [Dense tables](../experimental/token_replay/dense-results.md) |36 points,25755 profiling+1716 warmup requests, zero errors; roomy cache effect ~1.65pp, modest throughput | Same seed repeated runs; actual closed-loop mix differs. Counts are not every output-ID audit or GPU FLOPs. |
| E3 | [447 reference](../experimental/token_replay/evidence/dense-corpus-reference.json), [check_engine.py](../experimental/token_replay/check_engine.py) | All447 returned IDs/length finishes exact; existing single cached-versus-fresh follow-up checks output/logprob agreement | No universal lifecycle or numerical KV equality guarantee. Do not claim that no functional cache witness exists. |
| E4 | [HTTP overhead](../experimental/token_replay/evidence/dense-overhead-02.json) |32 fixed real prompts; matched generated outputs/cache work; c1/4/8 upper adjusted interval <1%; zero retractions | HTTP boundary, greedy, prewarmed; not full AgentX or a source-population guarantee. |
| E5 | [Phase boundary](../experimental/token_replay/evidence/phase-boundary-final.json) |37 runs: warmup HTTP requests drain before profiling | Does not remove warmup cost or equalize snapshot with full history. |
| E6 | [MoE report](../experimental/token_replay/evidence/moe-service-routing-summary.json) |308 real/synthetic requests per arm; fixed TP2/EP2; routing distributions differ; throughput interval spans zero | Representative rank timing covers9/308; no routing-to-time causal claim. |
| E7 | [Final delivery review](../experimental/token_replay/evidence/final-review-20261107.json) | Raw-data reconstruction and deterministic checks pass; analysis flag fix re-reviewed | Codex-only provisional; Claude unavailable. This is an earlier DELIVERY review, not this proposal's method review. |

Source freeze: InferenceX c2e804c4523d73162018fac18850f62a806387b2; harness b9772d220f86322055680a28736958c9f5b1fe4c; SGLang d611ff2171e95a88e6aac7c5b24cc224a3b5bbd5. Deployed wheel/source identities remain in the existing reproduction guide.

## Execution-semantic finding

The delivered wheel supports --request-count (wire credits, including children) and --num-sessions/--num-conversations (root admission budget). The InferenceX wrapper currently forwards duration only. A request cap can cut off future tree turns; a session cap still needs correct handling of roots already active at snapshot handoff. Warmup advances/recycles trajectories. Thus447 source rows do not imply447 profiling calls. Any future overhead extension must freeze the actual expanded invocation ledger and validate drain/termination. Use existing scheduler and finite-session support; do not create a replacement scheduler just to obtain a tidy count.
