# Replay overhead investigation and optimization decisions

The fixed-text, single-request acceptance run on GPU6 GPU2 passed the scoped3%
overhead target. Mean paired HTTP overhead was0.0861%, with a fixed-final95%
Student t interval of[-0.0321%,0.2044%]. The pinned asynchronous staging candidate
was slower for the small requests and remains experimental. Serving code was
not changed.

## Measurement correction

The earlier GPU0 measurement shared its GPU with another user's active
reinforcement-learning process, PID799584, already visible in the earlier
preflight. This prevents attributing its73.091% outlier to token replay.
GPU isolation, block aggregation and warmup protocol all changed in the new
run, so the improvement between the two runs is not an isolated causal effect
of removing that process. No other user's process was stopped.

The original first failed assertion recorded neither its arm nor mismatched
IDs. Its root cause cannot be reconstructed from that artifact. The new
request ledger saves every response and its context before checking it,
including ordinary/replay output IDs, full metadata, wall-clock start and
client elapsed time. Ordinary output drift is distinguished from replay
mismatch. Neither type occurred in this new run.

## Frozen diagnostic and observed result

The [specification](replay-diagnostic-20261112.spec.json) uses the existing
`run_campaign.py` guard, receipt, metric capture and process cleanup. GPU2 had
4MiB allocated before launch. The model, attention backend, seed42 and
32768-token KV capacity are fixed. The workload has a230-token prompt and32
output tokens. Each arm starts from a flushed cache; flushing and ledger
writes are outside the request timer in both arms.

Two warmup pairs precede20 measured pairs. Each arm contains four sequential
requests; arm order alternates. All20 pairs, including extrema, enter the
final interval. There is no early stopping or removal of slow samples. The
interval describes repeated execution of this fixed text under the paired t
assumptions, not a prompt-population estimate or an AgentX workload result.

| Check or measurement | Result |
| --- | --- |
| Recorded requests | 181 total;160 measured,80 in each arm |
| Ordinary and replay output | All measured output IDs match the captured reference |
| Warmup comparability | Both pairs valid |
| Measured cache/computation | Zero cached input tokens per request; equal input/output lengths; zero retractions |
| Mean HTTP overhead | +0.0861% |
| 95% interval | [-0.0321%, +0.2044%] |
| Largest paired HTTP overhead | +0.6314% |
| Mean paired server-reported overhead | +0.1189%, diagnostic only |
| Roomy continuation | 262 cached tokens against261 eligible prefix tokens; cold reference has0 |
| Cached/cold continuation outputs | Equal |

The [raw run](evidence/replay-diagnostic-20261112-run01/) contains the request
ledger, result, exact launch receipt, server information, GPU samples and
before/after metrics. The initial reference and first replay took207.8ms and
206.4ms respectively; both are retained outside the steady-state estimate.
The campaign stopped its own server after completion and GPU2 returned to4MiB.

## Asynchronous staging candidate

`TokenReplayBatch.from_requests` constructs two device tensors. Its ordinary
path showed two `cudaStreamSynchronize` calls. The candidate builds pinned CPU
tensors and submits nonblocking copies on the same CUDA stream. The profiler
confirmed removal of those two stream synchronizations. All staged sequences
and first-token replacement results passed explicit checks, including mixed
ordinary/replay rows and collection of temporary host objects.

The [component experiment](profile_replay_staging.py) runs the actual constructor
with a tensor factory proxy. Both arms include the same patch dispatch; pinned
allocation, CPU construction and H2D transfer are included. Five warmups and30
alternating pairs are used per case. Timings are component medians, include
experiment wrapper cost and are not serving overhead.

| Batch | Output length | Baseline ready time, us | Pinned async ready time, us | Change |
| --- | --- | ---: | ---: | ---: |
| 1 | 32 | 83.88 | 96.96 | +15.59% |
| 1 | 128 | 91.62 | 106.07 | +15.77% |
| 32 | 128 | 353.41 | 364.84 | +3.23% |
| 128 | 128 | 1145.72 | 1147.82 | +0.18% |
| 32 | 8192 | 12726.87 | 12636.90 | -0.71% |

The candidate does not improve the current small-request target. Removing a
synchronization does not by itself offset the extra host staging cost. The
[symmetric raw comparison](evidence/replay-diagnostic-20261112/staging-paired/)
contains all pairs and Chrome traces. The earlier exploratory comparison in
the parent directory used unequal wrapper overhead and is retained separately;
it is not the basis of the table.

The existing replacement kernel measures about1us per call. At32 output steps,
its device work is small relative to the roughly202ms ordinary request. Moving
the recorded-token gather earlier is possible because it is independent of
logits, but writing sampled output still depends on the sampler finishing.
A separate gather buffer, event and later copy would add work. No evidence
here supports that stream change or bypassing normal forward/sampling.

## Next execution gate

1. Retain the current serving implementation and the improved ledger. The old
   first-call drift remains an explicitly unlocated historical failure; a new
   mismatch must preserve its ordinary/replay arm and IDs before diagnosis.
2. Extend the fixed-text protocol to output lengths128/512 and concurrency4/8
   on an idle GPU, before adding AgentX trajectories. Preserve identical outputs,
   cache work, request counts and zero retractions between arms. Freeze each
   extension's counts before execution and keep the3% upper-interval gate.
3. Profile an extension only when it is inconclusive or exceeds that gate.
   Separate CPU tensor construction, H2D staging, kernel launch and kernel
   device duration. The current constructor experiment points to host data
   construction at large payloads; it does not establish a serving bottleneck.
4. If admission staging is material in that failing condition, evaluate a
   reusable pinned buffer with explicit lifetime through H2D completion. Compare
   both host-return and data-ready time, then rerun matched HTTP overhead and
   filter/merge/lifetime checks. Adopt only a repeated end-to-end improvement
   with all token/cache checks passing.

No concurrent AgentX or larger-model campaign was started by this diagnostic.
