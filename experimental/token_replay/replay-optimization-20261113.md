# Fixed-text replay validation and optimization

The GPU6 validation completed a nine-cell length and batch-size matrix, two
additional fresh-server restarts, a mixed normal/replay batch, and an
A/B/A cache-salt check. Every cell passed exact output and work checks, all
eligible KV prefixes were reused, and every familywise 95% HTTP overhead
interval stayed below 3%. This is fixed-text component evidence; it does not
measure AgentX trajectories or a prompt population.

## Frozen run

The run used GPU6 GPU2, an RTX 4090, the Qwen2.5-7B-Instruct GPTQ Int4 model,
SGLang 0.5.10, Triton attention, seed 42, a 230-token text prompt and KV
capacity 32768. The nine cells combine output lengths 32, 128 and 512 with
batch sizes 1, 4 and 8.
Each cell captures a cell-specific ordinary reference, then records two warmup
pairs and 20 measured alternating ordinary/replay pairs. Each arm starts after
an equal cache flush; the flush and JSON ledger write are outside the timer.
Every slow pair remains in the estimate.

The primary metric is replay HTTP batch wall time divided by ordinary HTTP
batch wall time minus one. The fixed 3% target is assessed with a Bonferroni
familywise 95% paired-t interval across the nine matrix cells. Each measured
pair must preserve output IDs and text, prompt and completion lengths, zero
cached prompt tokens, and zero retractions. These checks make the two arms do
the same generation and KV work. The matrix has 1560 measured request outputs
and 349440 measured output tokens. Its complete request ledger has 1926 rows,
including references, warmups, cache checks and isolation requests.

| Output tokens | Batch | Mean HTTP overhead | Familywise 95% interval | Eligible KV prefix reuse |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 1 | -0.2560% | [-0.8133%, 0.3012%] | 100% |
| 32 | 4 | +0.0490% | [-0.0644%, 0.1625%] | 100% |
| 32 | 8 | +0.1463% | [-0.0069%, 0.2994%] | 100% |
| 128 | 1 | +0.0414% | [-0.0063%, 0.0891%] | 100% |
| 128 | 4 | +0.0545% | [0.0178%, 0.0912%] | 100% |
| 128 | 8 | -0.1264% | [-0.7148%, 0.4620%] | 100% |
| 512 | 1 | +0.0342% | [0.0146%, 0.0537%] | 100% |
| 512 | 4 | +0.0800% | [-0.0238%, 0.1838%] | 100% |
| 512 | 8 | +0.2586% | [-0.1773%, 0.6946%] | 100% |

The smallest eligible prefix is 261 tokens and the largest 741 tokens. For each
cell, the cached continuation reused at least the entire eligible prefix; a
flushed continuation reused zero tokens and returned the same output. Mixed
requests of lengths 3, 7, 13 and 11 preserved forced and ordinary outputs,
and an ordinary request following the mixed batch matched its reference. The A/B/A
cache-salt witness reported cached-token counts 0, 0 and 229.

## Fresh-server and first-use behavior

The matrix server and two later servers used distinct empty Triton, Inductor
and CUDA compilation-cache directories. Each server ran the same 32-token,
batch-one cold suite, including a new output reference and 20 measured pairs.
The captured first replay output IDs and text match across all three server
starts.

| Server start | First ordinary request | First replay request | Warmed mean overhead | 95% interval (matrix familywise; cold pointwise) |
| --- | ---: | ---: | ---: | ---: |
| Matrix | 207.1 ms | 451.8 ms | -0.2560% | [-0.8133%, 0.3012%] |
| Cold02 | 207.4 ms | 425.8 ms | +0.0291% | [-0.1206%, 0.1789%] |
| Cold03 | 207.4 ms | 432.8 ms | -0.0141% | [-0.2863%, 0.2582%] |

The first replay request cost an additional 218-245 ms after an ordinary
reference on each fresh server, consistent with lazy compilation of the replay
kernel. This first-use latency is excluded from the warmed overhead intervals
and must remain visible when interpreting the 3% result. A deployment that
needs a low-latency first replay should prewarm the replay path before serving
latency-sensitive traffic and measure that startup cost separately.

## Failure and fix

The first matrix attempt is preserved in
[`evidence/complete-replay-validation-20261112-run01/`](evidence/complete-replay-validation-20261112-run01/).
It exposed `TypeError: unhashable type: 'list'` when a batched `extra_key`
cache-salt list reached the single-request KV cache key. A regression failed
before the fix and passed after it. The fix validates salt-list length,
replicates salts in parallel sampling order and selects the scalar salt for
each request. The focused test covers batched and single requests, parallel
sampling, scalar and null salts, and invalid list lengths. The SGLang input
normalization suite also passed all 23 tests. Separate GPU3 component checks
passed 200 seeded storage merge/reorder/lifetime rounds and the ordinary
fast-path, in-flight and rejoin checks.

The follow-up patch against SGLang base `1519acf` is
[`sglang-cache-salt-followup.patch`](sglang-cache-salt-followup.patch); its
application and source-hash parity are recorded in
[`patch-check.json`](evidence/complete-replay-validation-20261113-inputs/patch-check.json).
The four local commits each contain one function-level change. GPU6 served the
patched source whose SHA-256 matches the patch receipt.

## Optimization evidence

CUDA events around the Triton replacement kernel measured median device times
of 1.022, 1.063, 1.114 and 1.155 microseconds at batch sizes 1, 32, 128 and
512. These measure the whole kernel, not the cost of individual `tl.load`
operations.
Nsight Compute is installed on GPU6, but its hardware-counter query was denied
with `ERR_NVGPUCTRPERM`; instruction-level and memory-transaction counts were
therefore unavailable to this user.

A pinned asynchronous host-staging prototype removed two stream synchronizes
but increased ready time 15.6% for batch 1/length 32 and 15.8% for batch
1/length 128; other tested cases ranged from 0.18% slower to 0.71% faster. It
was not applied to serving. The measured kernel is about one microsecond while
complete requests take hundreds of milliseconds to several seconds, and the
matrix found no steady-state overhead problem near the 3% gate.

The proposed pre-sample token lookup was also tested as an isolated CUDA-graph
prototype. It launches the existing lookup on a side stream while a 151000-wide
`argmax` surrogate runs, then copies the prefetched values over the sampled
IDs. Thirty alternating pairs per batch preserved token IDs. The paired mean
device-time changes were -3.13% for batch 1 (95% interval [-7.09%, 0.83%]),
-0.05% for batch 4 ([-2.21%, 2.11%]) and -0.06% for batch 8 ([-2.35%, 2.23%]);
all intervals include zero. This synthetic sampler surrogate is not an
end-to-end SGLang result. It does not support changing the serving sampler.
The side-stream version also needs a separate output buffer and a post-sample
copy. The production replay kernel remains unchanged.

The [prototype source](profile_async_prefetch.py),
[raw timings](evidence/complete-replay-validation-20261113-inputs/async-prefetch-result.json)
and [paired audit](evidence/complete-replay-validation-20261113-inputs/async-prefetch-audit.json)
are retained with the run inputs. Nsight Compute is installed on GPU6, but its
hardware-counter query was denied with `ERR_NVGPUCTRPERM`; instruction-level
and memory-transaction counts were therefore unavailable. Before attempting
per-load rewrites, obtain GPU performance-counter access and profile the
first-use JIT path separately from steady-state execution.

The full raw data, frozen specifications and source snapshots are in
[`evidence/complete-replay-validation-20261113-run02/`](evidence/complete-replay-validation-20261113-run02/)
and
[`evidence/complete-replay-validation-20261113-inputs/`](evidence/complete-replay-validation-20261113-inputs/).
The independent auditor recomputes every paired interval from the request
ledger and verifies source hashes, IDs, cache work, mixed requests, isolation
and cross-start outputs; its result is
[`audit.json`](evidence/complete-replay-validation-20261113-run02/audit.json).
