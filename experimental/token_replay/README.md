# Recorded-token AgentX experiment

The implementation and claim boundaries are documented in the
[meeting report](../../inferencex-e2e/docs/token-replay-meeting-review.md).
The report distinguishes completed evidence from running or missing gates.
All workload selection and sampling seeds are42. Token extensions are local
experiments with `submission_valid=false`.

For a Chinese walkthrough of the implementation, evidence boundaries and source
reading order, open the [interactive learning page](learning/index.html) directly
in a browser, or read the [learning notes](tutorial.md). The page is self-contained
and works offline; source and evidence links use the surrounding repository.

## Code and environment

| Component | Frozen base or artifact |
| --- | --- |
| AgentX harness1.0.13 | Base `60a094f256b42f55fde1efb3d45672e1d9a44566`, tag `agentx-v1.0.13`; [patch](agentx-harness-v1.0.13.patch) through `b9772d2` |
| Deployed harness wheel | [wheel](evidence/agentx_harness-1.0.13-py3-none-any.whl), SHA-256 `9270c07b76981109a204a8c3b66cf706b501228ac194da3a9608e8a3a9676037`; build source `cedfe3f`; later harness changes are tests/docs |
| SGLang0.5.10 | Base `1519acf`; [patch](sglang-v0.5.10.patch) through `d611ff2` |
| Actual serving source | [residency-enabled overlay](evidence/serving-source-residency-20261106.tar.gz); two format-only differences are recorded in [AST parity receipt](evidence/serving-source-parity.json) |
| Serving environment | Python3.12, torch2.9.1+cu128, transformers5.3.0, triton3.5.1, FlashInfer0.6.7.post2 |
| Dense model | `/home/tegao/models/Qwen2.5-7B-Instruct-GPTQ-Int4` |
| MoE model | Qwen1.5-MoE-A2.7B-Chat BF16, revision `ec052fda178e241c7c443468d2fa1db6618996be`, TP2/EP2 |

[Model content manifest](evidence/model-content-manifest-20261106.json) records
SHA-256 for actual weights, model/tokenizer configuration and vocabulary files.
The existing Dense checkpoint has no verified source revision; its content
hashes, rather than an inferred Hugging Face revision, identify the artifact.

Apply each patch in a separate clean checkout at its stated base with
`git apply --check` followed by `git apply`. Use the archived wheel for byte-exact
harness reproduction. The serving overlay archive records the exact Python
source deployed for the reported runs. No shared environment upgrade is needed.

The public stochastic path additionally needs existing cuRAND headers on
`CPATH`; [the public specification](weka-original-04.spec.json) and
[actual kernel witness command](stochastic-witness-command.txt) record the exact
isolated include/cache configuration. Import-only checks do not establish this.

## Data and commands

The [acceptance archive](evidence/entry-evidence-20261106.tar.gz) contains the
corrected selected datasets and manifests. The pinned Exgentic revision is
`4b8ad4ab198438e5a170f9171c19c6a2cf7c1814`. Use `exgentic-dense-02`,
`exgentic-moe-02` and `exgentic-moe-synthetic-01`; older Dense data is diagnostic.
The public hash-only Weka revision is
`23f152f6f0f9399a85901b89a6458def0ef16729`. Its64K YaRN reference is separate
from the16K Dense causal comparison.

The natural KV-pressure L1 follow-up is frozen separately in
[natural-kv-l1-20261109.spec.json](natural-kv-l1-20261109.spec.json). It reuses
the Dense selection and identity hashes but leaves
`SGLANG_TEST_RETRACT`, `SGLANG_TEST_RETRACT_INTERVAL` and
`SGLANG_TEST_RETRACT_NO_PREFILL_BS` unset. Use a fresh GPU0 guard, a unique
output root and at most the two attempts declared by the spec. A natural L1
PASS requires a same-request `retraction_trigger` with `trigger=kv_full`, then
the complete reset/resume/output-ID chain; test-trigger or pause-trigger events
are not eligible. The 2026-11-09 guard found GPU0 occupied before launch, so
that run is recorded as INCOMPLETE in
`evidence/natural-kv-l1-20261109/guard-receipt.json`.
The local/gpu6 artifact hashes are pinned in
`natural-kv-l1-20261109.manifest.json`; the same manifest and source artifacts
are staged under `/home/tegao/infra-token-replay/` on gpu6.

These commands run on gpu6 after placing the checked-in scripts/specifications
and archived selected datasets beneath `/home/tegao/infra-token-replay`.
Output directories must not exist. Preserve all failed outputs and choose a new
run ID for retries. A fresh GPU guard runs before each server/engine launch;
GPU0 is Dense, GPU2/3 are MoE, and GPU1 is not allocated to this study.

```bash
python3 /home/tegao/infra-token-replay/run_campaign.py \
  --spec /home/tegao/infra-token-replay/weka-original-04.spec.json \
  --output /home/tegao/infra-token-replay/weka-reference-reproduction

python3 /home/tegao/infra-token-replay/run_campaign.py \
  --spec /home/tegao/infra-token-replay/dense-formal-01.spec.json \
  --output /home/tegao/infra-token-replay/dense-formal-reproduction
```

The Dense specification fixes all36 points and900-second admission intervals.
The runner invokes the original InferenceX `infx.bench agentic` command, captures
startup and phase evidence, drains requests and shuts down only its own process
groups. It fails on request errors/skips/cancellations. Do not compare different
data sources as paired modes or shorten admission intervals while retaining the
formal-study label.

MoE commands and explicit environment variables are frozen in
[the perturbation stage specification](moe-perturbation-stages-01.spec.json).
`moe_workload.py` performs normal forward/sampling plus token replacement,
flushes between sessions and checks exact output IDs. The off, on and profile
specifications separate performance, routing and profiler diagnostics. Native
clients and analysis use the isolated Python3.12/3.11 environments; system
Python3.8 is only used for compatible orchestration.

## Evidence processing

Use `summarize_campaign.py` then `compare_dense.py` for phase-separated Dense
results. Retain the actual closed-loop request mix, source-conditioned cached
tokens, drain denominator, evictions, retractions and sampled residency.
Non-evictable token usage is not total resident cache size. Five-second memory
samples are not instantaneous allocation peaks.

The [complete36-point Dense tables](dense-results.md) link the72-row phase CSV
and source-level summaries. `export_dense_table.py` exports those phase rows.
`audit_phase_boundary.py` checks actual warmup/profiling HTTP request intervals;
it does not measure GPU event overlap or remove warmup's initial-cache effect.
The separate all447-request `audit_dense_corpus.py` run checks exact output IDs
from session starts. Performance-run usage checks establish lengths and cache
counts, not retrospective token-ID equality for every performance repetition.

For MoE, run `audit_moe_recorder.py` on both content directories, writing
`coverage-audit.json`, then `summarize_moe.py`. The audit checks both replicas,
input IDs/positions, expert placement and per-layer assignment counts. In this
sequential overlap configuration the final output token is forwarded once before
its result is discarded; recorder decode inputs therefore cover full outputs.
Global histograms are replicated and must not be summed across ranks.

Run `summarize_rank_profiles.py` followed by `compare_rank_profiles.py` for the
nine selected full requests per content. Compute/other and communication unions
can overlap; neither their sum nor summed ranks is elapsed wall time.
`compare_recorder.py` separately reports paired service cost and workload cost
including flush/dump/request writes, with initialization recorded independently.

Raw evidence beneath `evidence/` uses Git LFS. Preserve specifications,
receipts, model/data revisions, excluded source windows and negative runs with
the final tables. Codex-only positive reviews remain provisional when Claude
review is unavailable; deterministic checks support only their tested scope.

`verify_delivery.py` expects the flat campaign root used on gpu6. The checked-in
evidence keeps large raw directories in separate archives, so a local replay of
the delivery check first stages all required archives into a temporary root:

```bash
cd experimental/token_replay
review_root=$(mktemp -d)
for archive in \
  evidence/dense-formal-evidence-20261106.tar.gz \
  evidence/entry-evidence-20261106.tar.gz \
  evidence/dense-preflight-overhead-20261106.tar.gz \
  evidence/moe-recorder-paired-evidence-20261106.tar.gz \
  evidence/storage-doc-witness-evidence-20261106.tar.gz \
  evidence/weka-reference-evidence-20261106.tar.gz \
  evidence/moe-routing-rank-evidence-20261106.tar.gz; do
  tar -xzf "$archive" -C "$review_root"
done
python3 verify_delivery.py \
  --root "$review_root" \
  --output "$review_root/delivery-verification.json"
```

The natural guard receipt is compact evidence and does not replace the seven
archived B0 evidence packages required by `verify_delivery.py`. The new natural
specification and validator are source artifacts; they do not inherit review
acceptance from the prior source/wheel freeze.
