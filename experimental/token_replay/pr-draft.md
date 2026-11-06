[AgentX] Preserve recorded token trajectories / [AgentX] 保持记录的 token 轨迹

Generated replies can diverge from the fixed prompt of the next recorded turn.
This change adds explicit generated and recorded-token modes to the original
AgentX scenario, so both arms use the same prompts, response lengths, scheduler,
snapshot warmup and session isolation. The recorded arm retains normal model
forward/sampling and replaces the sampled ID before the next forward.

The implementation spans the pinned AgentX harness 1.0.13 patch and SGLang 0.5.10
patch. It validates full-trace budgets, preserves one-token warmup semantics and
uses cache salts without modifying prompt tokens. Local token experiments are
marked `submission_valid=false`; the default public-data route remains available.

The original InferenceX entry completed 240 requests: 198 profiling and 42 warmup,
with zero errors and exact wire/output token matches. Actual event records cover
parent/child salts, joins, nonzero waits, context reset and snapshot handoff.
An identical-prompt A/B/A witness also exposed and verified the fix for a missing
SGLang scheduler salt assignment. See the [meeting evidence](../../inferencex-e2e/docs/token-replay-meeting-review.md)
and [raw acceptance archive](evidence/entry-evidence-20261106.tar.gz).

A fixed32-prompt matched-output HTTP experiment met the scoped3% overhead target
at concurrency1/4/8. The full308-request MoE recorder-off comparison found a
synthetic-relative-to-real throughput difference of+0.2336% (95% interval
[-0.0285%, +0.4958%]); it does not establish a throughput effect.

The full36-point AgentX matrix completed25755 profiling and1716 warmup requests
with zero errors. Roomy-cache source-standardized cache hits increase by about
1.65–1.66 percentage points; throughput point estimates increase by
0.088%/0.569%/0.847% at concurrency1/4/8. Pressure concurrency4/8 throughput
intervals span zero. The [complete tables](dense-results.md) retain phase costs,
request mix, per-user/per-GPU throughput, latency, residency and retractions.
An independent from-start reference checks exact output IDs for all447 qualified
Dense requests; [deterministic delivery checks](evidence/delivery-verification.json)
pass. Performance-run length checks are not retrospective per-request ID audits.

All37 audited Dense/public runs drain warmup HTTP requests before profiling;
warmup still contributes substantial cache work and changes starting state.
MoE routing coverage, nine-request dual-rank diagnostics and three paired recorder
controls completed; recorder service overhead is about21%, so content-performance
claims use recorder-off measurements. The acceptance fixture is synthetic and is not an original Weka score;
Exgentic content is re-tokenized for the target model, with processed timing and
no invented DAG. The available review is Codex-only: Claude-review was unavailable.
No completed cross-family review or general performance improvement is claimed.
The [final independent review](evidence/final-review-20261107.json) reconstructed
the raw Dense results and verified the scoped analysis fix without blocking findings.

<details><summary>Implementation and validation evidence</summary>

- [Reproduction guide](README.md) and [harness patch](agentx-harness-v1.0.13.patch), against tag `agentx-v1.0.13`.
- [SGLang patch](sglang-v0.5.10.patch), against base `1519acf`.
- [Exact deployed source archive with cache-residency metrics](evidence/serving-source-residency-20261106.tar.gz).
- [Dense matrix specification](dense-formal-01.spec.json), frozen after successful
  roomy/pressure preflights, with actual eviction in the pressure condition.
- [MoE performance specification](moe-workload-off-01.spec.json), fixed 308-request
  real/synthetic order, TP2/EP2 BF16, EPLB off and recorder absent.

The request-log acceptance runs are separate from performance measurements.
The public Weka reproduction is a separate descriptive baseline, not a numerator
or denominator for cross-dataset improvement percentages.

</details>

<details><summary>AI model disclosure</summary>

Codex, GPT-6 family, implemented and integrated the changes. Fresh Codex agents
performed independent protocol/campaign/delivery reviews and the documented MoE
environment witness. Their exact runtime model variants could not be verified. No Claude
review was available for this delivery.

</details>

<details><summary>中文</summary>

将完整输入／输出 token 接回原 AgentX 场景，保留原调度、snapshot warmup、
父子代理及会话隔离。逐 token 重放仍执行正常 forward 和 sampling，并在下一次
forward 前替换记录 ID。本地扩展实验明确标记为非官方提交结果。

原 InferenceX 入口已完成 240 个请求的精确 ID 与事件验收，零请求错误。
MoE 完整路由覆盖、代表请求双卡诊断与三对 recorder 控制已完成。
Dense 的36点矩阵已完成，25755个正式请求和1716个预热请求零错误。
宽裕KV下缓存命中率约提高1.65–1.66个百分点，吞吐增幅点估计小于1%；
压力档并发4／8未建立吞吐差异。已核验37次运行的预热HTTP请求先于正式请求结束，
但预热成本和初始缓存影响仍需单独报告。Claude审阅缺失，不声称双审通过。

</details>
