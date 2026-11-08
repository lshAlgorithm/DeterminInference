# Natural KV-Pressure L1 Plan

Date: 2026-11-09. This is a plan and freeze record; it is not experiment
evidence and does not claim natural-pressure coverage.

## Scope

Use the existing Dense Qwen2.5-7B-Instruct-GPTQ-Int4 corpus and lifecycle
observer. Keep seed 42, the selected source rows and all model, serving source,
observer and wheel hashes from the frozen lifecycle identity. The ordinary
source and replay output vectors remain immutable. The spec is stored at
`experimental/token_replay/natural-kv-l1-20261109.spec.json`.

Only the scheduler's natural KV-full condition is eligible. The test-retraction
environment variables must be unset, pause/continue endpoints are unused, and
no artificial sleep or request abort may be used to create L1. Every request ID
must use the `natural-b1-` namespace and the output root must be unique.

## Budget and order

1. Run a fresh GPU0 guard on gpu6 and record the exact `nvidia-smi` snapshot.
2. Start the pinned Dense server with context 16384, max total tokens 131072,
   max running requests 16, Triton attention and seed 42.
3. Admit the frozen mixed replay/ordinary schedule and collect raw wire and
   observer events.
4. Allow at most one predeclared escalation, for at most two attempts total;
   the combined GPU allocation, startup, drain and teardown must stay within
   0.5 GPU-hour.
5. Stop immediately on a mismatch, error, cancellation leak or deadline; keep
   the raw output and classify the run as FAIL or INCOMPLETE.

## Acceptance

Natural L1 PASS requires one request ID to show partial output, a
`retraction_trigger` with `trigger=kv_full`, `reset_for_retract`, resumed extend,
continued decode and length finish. The consumed prefix, resumed next output
ID, final IDs, length and finish reason must match the greedy reference exactly.
No natural trigger or incomplete same-request chain is UNCOVERED. A failed GPU
guard or an unstarted attempt is INCOMPLETE. Neither result changes the current
fixed-32-prompt HTTP overhead claim or activates B2.

## Delivery

Retain the spec hash, source/wheel/model hashes, seed, request IDs, command,
GPU guard, raw logs and validator output. Run `check_natural_lifecycle.py`,
`git diff --check`, Python compilation and the existing seven-archive
`verify_delivery.py`. Codex review remains provisional; Claude review is
unavailable, so no dual-review acceptance is recorded.
