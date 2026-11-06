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

The fixed real-content AgentX performance matrix and full MoE diagnostics are not yet
complete. The acceptance fixture is synthetic and is not an original Weka score;
Exgentic content is re-tokenized for the target model, with processed timing and
no invented DAG. The available review is Codex-only: Claude-review was unavailable.
No completed cross-family review or workload-wide performance improvement is claimed.

<details><summary>Implementation and validation evidence</summary>

- [SGLang patch](sglang-v0.5.10.patch), against base `1519acf`.
- [Exact deployed source archive](evidence/serving-source-20261106.tar.gz).
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
performed independent protocol/campaign reviews and the documented MoE environment
witness. Their exact runtime model variants could not be verified. No Claude
review was available for this delivery.

</details>

<details><summary>中文</summary>

将完整输入／输出 token 接回原 AgentX 场景，保留原调度、snapshot warmup、
父子代理及会话隔离。逐 token 重放仍执行正常 forward 和 sampling，并在下一次
forward 前替换记录 ID。本地扩展实验明确标记为非官方提交结果。

原 InferenceX 入口已完成 240 个请求的精确 ID 与事件验收，零请求错误。
完整 Dense 性能矩阵和 MoE 诊断尚未完成，短跑不代表会议目标全部解决；
Claude 审阅缺失，也不声称双审通过。

</details>
