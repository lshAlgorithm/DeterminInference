# B1 Interface Notes

Read-only source/archive inspection: 2026-11-07 by `/root/refine_evidence_gaps`; exact runtime model identifier unavailable. No source mutation or GPU work. These are source facts and planned witnesses, not executed B1 evidence.

| Interface | Pinned source | Observed semantics |
| --- | --- | --- |
| Deterministic retraction | [scheduler.py](../../sglang/python/sglang/srt/managers/scheduler.py#L2574) | Test interval and KV-full paths both call batch.retract_decode(); log trigger separately. |
| Resumed input construction | [schedule_batch.py](../../sglang/python/sglang/srt/managers/schedule_batch.py#L940) | fill_ids equals origin_input_ids plus retained output_ids. |
| Reset | [schedule_batch.py](../../sglang/python/sglang/srt/managers/schedule_batch.py#L1187) | reset_for_retract preserves text-token output_ids and increments retraction state. |
| Abort | [scheduler.py](../../sglang/python/sglang/srt/managers/scheduler.py#L3266) | /abort_request uses RID prefix matching; running request may decode again for cleanup; wait for abort terminal response. |
| Per-request receipt | [tokenizer_manager.py](../../sglang/python/sglang/srt/managers/tokenizer_manager.py#L1563) | Native meta_info includes request ID and total_retractions. |
| Upstream example | [test_retract_decode.py](../../sglang/test/registered/scheduler/test_retract_decode.py#L24) | Shows test controls but uses other model/data; do not run as this project's B1. |

Source paths are sibling-checkout links; archives are the portable authority. The scheduler in `experimental/token_replay/evidence/serving-source-residency-20261106.tar.gz` matches the local scheduler byte-for-byte according to the inspection. The older serving-source-parity.json predates the final residency overlay and must not override the final archive.

Existing check_engine.py covers kernel/mixed/uneven metadata cases; check_live_lifecycle.py covers mixed 3/7/13 replies and budget rejection. Neither currently proves replay retraction/resume or explicit cancellation. B1 needs a new opt-in event observer and dedicated driver, with reviewed function-scoped edits before any GPU execution. The exact protocol and budget are in [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md).

B2 statistics source: [measure_overhead.py](../experimental/token_replay/measure_overhead.py). n=3 uses 6.205347, n=5 uses 3.495405; an n=3 interval containing or touching 0.03 continues to five; never add post-hoc looks. The adjustment covers two looks under t assumptions, not all concurrencies or the source population.
