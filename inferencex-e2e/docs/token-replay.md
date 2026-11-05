# Token-exact trajectory replay (unofficial)

## Source boundary

The supplied archive identifies source snapshot
`cf2e11e7d1680a9e60faf64d5f6d67b17237e690`. InferenceX's
`infx/bench/agentic/replay.py::replay_argv` assembles an `agentx profile`
command; it does not execute model decoding. Its requirements pin
`agentx-harness==1.0.13`. The matching harness tag resolves to
`60a094f256b42f55fde1efb3d45672e1d9a44566`.

In that harness, `src/aiperf/dataset/loader/weka_trace_models.py` defines
request lengths, block hash IDs, content types, and timing fields. It does
not define original prompt or response token IDs. `weka_synth_buf.py`
reconstructs synthetic content. The `DELTAS_WITH_RESPONSES` context mode
in `src/aiperf/common/enums/enums.py` explicitly discards live responses
when assembling subsequent requests. These sources support the concern
that generated response tokens need not match the next recorded prompt.

The harness also has staged trajectory warmup and optional accelerated
cache-pressure warmup in `src/aiperf/timing/strategies/agentic_replay.py`.
This alone does not establish the meeting's claim of a separate background
thread repairing KV state during measured decoding. Do not remove all
warmup on that assumption: model/CUDA-graph warmup and initializing a
trajectory at a sampled midpoint are distinct from repairing token drift.

## Intended implementation

1. Accept original, tokenizer-specific prompt and response token IDs.
   Preserve full prompt IDs at each turn, including chat-template tokens,
   tool results, and any context truncation. Reject hash-only input for
   token-exact replay; never describe generated filler as recovered text.
2. Carry the expected response IDs to the serving engine once per request.
   Stage IDs before decoding. Replace the sampled ID on-device before
   the scheduler, output bookkeeping, and the following forward pass
   consume it. Keep ordinary model forward and sampling work in the
   benchmark; skipping either changes what is measured.
3. Use the actual decode position rather than a global mutable callback
   counter. Requests can be batched, reordered, paused, and resumed.
   Stop after the recorded response length, including when an internal
   recorded ID is EOS. Validate token IDs against the served vocabulary.
4. Keep unsupported execution modes explicit. Speculative decoding needs
   acceptance and draft-state treatment; ordinary one-token replacement
   must not silently claim speculative support. Log probabilities must
   correspond to the emitted token or the request must reject them.
5. Replay subsequent turns without repair requests. Compare the emitted
   IDs and next-turn cache reuse against a reference run. Preserve
   trace delays and dependencies when integrating the AgentX workload.

For autoregressive decoding, the newly selected token normally has no KV
entry yet; its KV is produced by the next forward pass. Replacing only
the visible HTTP response is too late. Replacing its embedding alone
also leaves scheduler token history and prefix-cache keys inconsistent.
Precomputing embeddings is therefore not a prerequisite: staging token
IDs and using the model's normal embedding lookup preserves the usual
execution path.

## Verification and evidence still required

- Exercise mixed forced and ordinary requests, batch reorder/removal,
  prompt prefill, EOS, and variable response lengths.
- Run a real model on `gpu6`; compare every emitted ID and next-turn
  prefix reuse. A client mock or a tensor-only test cannot prove this.
- Measure baseline versus forced replay with identical prompt/response
  lengths, concurrency, warmup, and seed 42. Report staging cost separately
  from steady-state decode overhead; do not infer a speedup from correctness.
- Use real-token traces to study MoE expert routing. Hash-derived filler
  cannot establish representative routing or load balance. A dense model
  correctness test does not validate this claim.

This document records the implementation contract, not a completed or
performance-validated feature. Existing official benchmark recipes remain
unchanged until the replay path has serving evidence.
