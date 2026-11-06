# Alignment with the November 5 meeting

The work implements the central token-replacement mechanism and validates
it on a real SGLang server. It does **not** yet complete the meeting's
benchmark-methodology objective. The current deliverable is a working,
optimized prototype plus an experimental harness route; it is not a
validated repair of the original InferenceX AgentX scenario.

## Motivation and acceptance

The meeting's objective is to compare inference systems under the same
recorded agent trajectory, so generated-token drift does not change the
prefixes, KV reuse, recomputation, or work being measured. Token-level
replacement is the intervention. Reliable KV behavior and low additional
measurement overhead are the outcomes to establish. Real content is also
needed before interpreting MoE expert routing as representative.

The available meeting document is an AI-generated summary, including a
proposal to precompute response embeddings and a claim about an extra
KV-warmup thread. Those details require checking against the actual
serving path and benchmark implementation.

| Meeting requirement | Current evidence | Assessment |
| --- | --- | --- |
| Replace each generated token before the next decode | SGLang `sampler.py:172` applies replay after normal sampling, before logprobs, TP synchronization, and return. The kernel writes the recorded ID in place. | Implemented in SGLang 0.5.10. |
| Align the token history represented by KV and reuse it next turn | Real 24-token replay, 34 cached tokens on the follow-up, and cached/fresh output agreement with logprob tolerance 0.05. | Supported for a short dense-model test. No workload-level retention, eviction, or recomputation evidence. |
| Keep replay overhead low | Packed storage and metadata-only batch mutations; ordinary-only batches restore the no-replay path; kernel/device and small whole-request timings recorded. | Component-level evidence. Not an original-benchmark performance comparison. |
| Avoid additional cache-repair interference | The new client sends recorded turns without repair requests. | No original-versus-new warmup/repair A/B. Existing snapshot warmup is not proved unnecessary. |
| Verify hash-only public traces and implications | Weka request schema lacks original token IDs; a bounded public-data prefix agrees. | Schema and sampled-prefix confirmation; no full-corpus audit or recovery of original content. |
| Use meaningful real-token agent workloads, including MoE concerns | Real tokenizer IDs from a newly generated two-turn Qwen2.5 dense-model example. | Mechanism fixture only. No captured agent workload or MoE/multi-GPU validation. |
| Improve the supplied InferenceX replay path and prepare an important PR | New `token-replay` command, Mooncake raw-payload exporter, SGLang patch, tests, local commits. | Original AgentX scenario/recipes are unchanged; no PR opened. Fork/join dependencies are not represented by the new trace format. |

## What the implementation changed

- Added `replay_token_ids` validation and an OpenAI completion adapter.
- Staged recorded IDs on GPU and indexed them using the actual decode
  position. Normal model forward and sampling remain in the path.
- Replaced longest-response padding with packed storage. Filtering and
  merging copy request metadata while retaining the backing tensors.
- Dropped replay state when the last forced request leaves a batch.
- Added a native exact-ID client and an exporter using the existing
  harness Mooncake raw-payload loader.
- Exercised real GPU mutation/lifetime cases, a real model, concurrent
  HTTP/SSE, and a two-request `aiperf profile` run with raw artifacts.

The code stages token IDs, **not precomputed embeddings**. That differs
from the literal proposal in the minutes: the next normal forward still
performs its usual embedding lookup. This keeps scheduler history,
prefix keys, and the next model input consistent without an extra
embedding-input path. No experiment compares the two implementation
choices, so this is a justified design choice rather than a measured
superiority claim.

## Claims that must not be conflated

The 32 MiB to 351 KiB storage reduction, 213.78 to 3.37 ms staging change,
and 252.79 to 18.65 microsecond merge change compare **our padded prototype
with our compact prototype** on a seeded, skewed-length fixture. They do
not quantify improvement over the original InferenceX implementation.
Staging is a single observation per case.

The approximately 1 microsecond replacement timing uses CUDA events over
100 kernels in a CUDA graph. The production call in `TokenReplayBatch.apply`
still has host-dispatch cost. It is not a measurement of total per-token
end-to-end overhead. The six measured short requests per condition do
not establish a workload-level overhead bound or throughput improvement.

The two-request harness check verifies loader, payload, HTTP/SSE, ordering,
and export behavior. It uses `aiperf --custom-dataset-type mooncake-trace`,
not the original `agentx profile --public-dataset ...` scenario. There was
no harness warmup, and cache usage reporting was not enabled. It cannot
establish removal of the original replay's performance bias.

The actual harness source documents pre-profiling snapshot/cache-pressure
warmup. We have not identified a separate background thread that repairs
KV during measured decoding. Starting at a sampled midpoint can still
require initial cache population even with token-exact replay. Therefore
all warmup must not be classified as avoidable repair work.

## Next work that directly tests the motivation

1. Reproduce the original AgentX route on a captured agent trace that
   includes prompt/response IDs, tool turns, timing, and relevant branch
   dependencies. Identify exactly which warmup or repair operations occur.
2. Add the token-replay intervention to that same workload route. Preserve
   initial-cache conditions, model, seed, arrival pattern, concurrency,
   token lengths, and normal engine warmup across comparisons.
3. Compare original replay and token-exact replay against a faithful
   from-start trajectory reference. Measure cached/new prefill tokens,
   retention/eviction and recomputation under pressure, extra requests,
   TTFT/TPOT, per-user and per-GPU throughput, and replay overhead.
4. Only use a MoE run with representative captured content to claim an
   effect on expert routing/load balance. If that is outside the first PR,
   state it explicitly as follow-up work.

The original methodology-level goal remains incomplete until those
comparisons support it. The completed engineering work is useful and
aligned, but it should not be presented as proving that InferenceX now
measures system performance more accurately.
