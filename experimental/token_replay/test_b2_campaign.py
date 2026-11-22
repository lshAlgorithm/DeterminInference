from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from b2_campaign import (
    DEFAULT_TEARDOWN_RESERVE_SECONDS,
    MAX_BUDGET_SECONDS,
    CampaignError,
    _CampaignRunner,
    parse_gpu0_memory,
    run_campaign,
)


RUNNER_CODE = """
import json
import os
from pathlib import Path

phase = os.environ["B2_PHASE"]
arm = os.environ["B2_ARM"]
phase_id = os.environ["B2_PHASE_ID"]
occurrences_manifest = json.loads(os.environ["B2_EXPECTED_OCCURRENCES_JSON"])
work = phase != "warmup"
occurrences = []
if work:
    for identity in occurrences_manifest:
        occurrence_id = identity["occurrence_id"]
        record = {
            **{key: identity[key] for key in (
                "source_file", "source_row", "phase", "logical_root_id",
                "tree_occurrence_id", "turn", "reset_ordinal", "cache_salt_class",
            )},
            "occurrence_id": occurrence_id,
            "request_id": phase_id + "-" + occurrence_id,
            "output_ids": [41, 42],
            "output_length": 2,
            "finish_reason": "stop",
            "cache_hit_tokens": 2,
            "new_prefill_tokens": 3,
        }
        if arm == "replay" and os.environ.get("TEST_MISMATCH_OUTPUT") == "1":
            record["output_ids"] = [99]
            record["output_length"] = 1
        if arm == "replay" and os.environ.get("TEST_MISMATCH_CACHE") == "1":
            record["cache_hit_tokens"] = 3
        occurrences.append(record)

measurement_seconds = None
if phase == "calibration":
    measurement_seconds = 1.0 if arm == "generated" else 1.01
elif phase == "measurement":
    pair_number = int(phase_id.split("-")[1])
    overheads = [0.0, 0.01, 0.02, 0.01, 0.01]
    measurement_seconds = (
        1.0 if arm == "generated" else 1.0 + overheads[pair_number - 1]
    )

count = len(occurrences)
result = {
    "schema_version": 1,
    "status": "PASS",
    "phase": phase,
    "arm": arm,
    "run_id": phase_id + "-run",
    "loaded_sessions": json.loads(os.environ["B2_SELECTED_SESSIONS_JSON"]),
    "profiling_root_admissions": (
        json.loads(os.environ["B2_ROOT_ADMISSIONS_JSON"]) if work else []
    ),
    "drain": {
        "sent": count,
        "completed": count,
        "errors": 0,
        "skips": 0,
        "cancellations": 0,
        "pending": 0,
    },
    "occurrences": occurrences,
}
if measurement_seconds is not None:
    result["measurement_seconds"] = measurement_seconds
Path(os.environ["B2_RESULT_PATH"]).write_text(json.dumps(result), encoding="utf-8")
"""

MARKER_CODE = """
import os
from pathlib import Path

Path(os.environ["B2_CAMPAIGN_OUTPUT_DIR"], os.environ["TEST_MARKER"]).write_text(
    "ran", encoding="utf-8"
)
"""


class B2CampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base = Path(self.temp_dir.name)
        self.plan_path = self.base / "plan.json"
        self.output_dir = self.base / "run"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _command(self, code: str, *, env: dict[str, str] | None = None) -> dict:
        return {
            "argv": [sys.executable, "-c", code],
            "timeout_seconds": 5,
            "env": env or {},
            "cwd": str(self.base),
        }

    def _write_plan(
        self,
        *,
        budget: int = 5400,
        reserve: int = DEFAULT_TEARDOWN_RESERVE_SECONDS,
        gate_seconds: dict[str, float] | None = None,
    ) -> dict:
        gate_path = self.base / "runtime-gate.json"
        sessions = ["007.json", "001.json", "017.json", "015.json"]
        roots = [
            {"source_file": session, "logical_root_id": session.removesuffix(".json")}
            for session in sessions
        ]
        occurrence_manifest = []
        for index in range(53):
            session_index = index % len(sessions)
            session = sessions[session_index]
            occurrence_manifest.append(
                {
                    "occurrence_id": f"occ-{index + 1:03d}",
                    "source_file": session,
                    "source_row": index,
                    "phase": "profiling",
                    "logical_root_id": roots[session_index]["logical_root_id"],
                    "tree_occurrence_id": f"tree-{index + 1:03d}",
                    "turn": 0,
                    "reset_ordinal": 0,
                    "cache_salt_class": "test",
                }
            )
        frozen_plan_path = self.base / "frozen-plan.json"
        frozen_plan = {
            "schema_version": 1,
            "selected_sessions": sessions,
            "requests_by_arm": {
                arm: [
                    {
                        "request_id": f"b2-{arm}-{item['occurrence_id']}",
                        **{
                            key: item[key]
                            for key in (
                                "source_file",
                                "source_row",
                                "phase",
                                "logical_root_id",
                                "tree_occurrence_id",
                                "turn",
                                "reset_ordinal",
                                "cache_salt_class",
                            )
                        },
                    }
                    for item in occurrence_manifest
                ]
                for arm in ("capture", "generated", "replay")
            },
        }
        frozen_plan_path.write_text(json.dumps(frozen_plan), encoding="utf-8")
        runner_receipts = {}
        request_logs = {}
        gate_seconds = gate_seconds or {
            arm: 1.0 for arm in ("capture", "generated", "replay")
        }
        for arm in ("capture", "generated", "replay"):
            receipt_path = self.base / f"{arm}-receipt.json"
            log_path = self.base / f"{arm}-request.jsonl"
            receipt_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "arm": arm,
                        "gpu_active_seconds": gate_seconds[arm],
                    }
                ),
                encoding="utf-8",
            )
            log_path.write_text("runtime ledger\n", encoding="utf-8")
            runner_receipts[arm] = {
                "path": receipt_path.name,
                "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            }
            request_logs[arm] = [
                {
                    "path": log_path.name,
                    "sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
                }
            ]
        gate_path.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "evidence_scope": "runtime",
                    "runtime_artifacts_verified": True,
                    "selected_sessions": sessions,
                    "profiling_root_admissions": roots,
                    "expected_occurrence_multiplicity": len(occurrence_manifest),
                    "event_counts": {
                        arm: len(occurrence_manifest)
                        for arm in ("capture", "generated", "replay")
                    },
                    "provenance": {
                        "schema_version": 1,
                        "frozen_plan": {
                            "path": frozen_plan_path.name,
                            "sha256": hashlib.sha256(
                                frozen_plan_path.read_bytes()
                            ).hexdigest(),
                        },
                        "runner_receipts": runner_receipts,
                        "request_logs": request_logs,
                    },
                }
            ),
            encoding="utf-8",
        )
        plan = {
            "schema_version": 1,
            "campaign_id": "test-b2-campaign",
            "hard_budget_seconds": budget,
            "teardown_reserve_seconds": reserve,
            "gpu": {"physical_index": 0, "max_used_mib": 500},
            "selected_sessions": sessions,
            "profiling_root_admissions": roots,
            "occurrence_ids": [item["occurrence_id"] for item in occurrence_manifest],
            "occurrence_manifest": occurrence_manifest,
            "correctness_gate": {
                "path": str(gate_path),
                "sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest(),
            },
            "setup": self._command(MARKER_CODE, env={"TEST_MARKER": "setup.marker"}),
            "teardown": self._command(
                MARKER_CODE, env={"TEST_MARKER": "teardown.marker"}
            ),
            "phases": {
                "capture": self._command(RUNNER_CODE),
                "warmup": self._command(RUNNER_CODE),
                "calibration_pair": {
                    "generated": self._command(RUNNER_CODE),
                    "replay": self._command(RUNNER_CODE),
                },
            },
            "measurement_pairs": [],
        }
        plan["teardown"]["timeout_seconds"] = min(3, reserve)
        orders = [
            ["generated", "replay"],
            ["replay", "generated"],
            ["generated", "replay"],
            ["replay", "generated"],
            ["generated", "replay"],
        ]
        for index, order in enumerate(orders, start=1):
            plan["measurement_pairs"].append(
                {
                    "pair_id": f"pair-{index:02d}",
                    "order": order,
                    "commands": {
                        arm: self._command(RUNNER_CODE)
                        for arm in ("generated", "replay")
                    },
                }
            )
        self.plan_path.write_text(json.dumps(plan), encoding="utf-8")
        return plan

    def _refresh_gate_hash(self) -> None:
        plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
        gate_path = Path(plan["correctness_gate"]["path"])
        plan["correctness_gate"]["sha256"] = hashlib.sha256(
            gate_path.read_bytes()
        ).hexdigest()
        self.plan_path.write_text(json.dumps(plan), encoding="utf-8")

    def test_default_is_validate_only_and_never_probes_gpu(self) -> None:
        self._write_plan()

        result = run_campaign(
            self.plan_path,
            self.output_dir,
            gpu_probe=lambda: self.fail("validate-only must not probe the GPU"),
        )

        self.assertEqual(result["status"], "VALIDATED_NO_EXECUTION")
        self.assertFalse(result["arms_launched"])
        self.assertFalse(self.output_dir.exists())

    def test_gpu_guard_blocks_setup_and_records_no_teardown_run(self) -> None:
        self._write_plan()

        result = run_campaign(
            self.plan_path,
            self.output_dir,
            execute=True,
            gpu_probe=lambda: "0, 600\n1, 0\n",
        )

        self.assertEqual(result["status"], "BLOCKED_GPU_GUARD")
        self.assertEqual(result["gpu"]["used_mib_preflight"], 600)
        self.assertFalse((self.output_dir / "setup.marker").exists())
        self.assertEqual(result["teardown"]["status"], "NOT_RUN")
        self.assertFalse((self.output_dir / "setup").exists())

    def test_success_validates_pairs_expands_ci_and_writes_receipts(self) -> None:
        self._write_plan()

        result = run_campaign(
            self.plan_path,
            self.output_dir,
            execute=True,
            gpu_probe=lambda: "0, 4\n1, 23801\n",
        )

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["paired_interval"]["pair_count"], 5)
        self.assertEqual(result["paired_interval"]["decision"], "PASS")
        measurement_pairs = [
            pair for pair in result["pairs"] if pair["pair_id"].startswith("pair-")
        ]
        self.assertEqual(len(measurement_pairs), 5)
        self.assertTrue(all(pair["status"] == "PASS" for pair in measurement_pairs))
        self.assertEqual(result["teardown"]["status"], "PASS")
        self.assertTrue((self.output_dir / "setup.marker").is_file())
        self.assertTrue((self.output_dir / "teardown.marker").is_file())
        arm_order = [Path(item["path"]).parent.name for item in result["stages"]]
        self.assertLess(
            arm_order.index("pair-01-generated"), arm_order.index("pair-01-replay")
        )
        self.assertLess(
            arm_order.index("pair-02-replay"), arm_order.index("pair-02-generated")
        )
        self.assertEqual(
            json.loads((self.output_dir / "campaign-receipt.json").read_text())[
                "status"
            ],
            "PASS",
        )

    def test_runtime_gate_must_match_campaign_root_admissions(self) -> None:
        self._write_plan()
        gate_path = self.base / "runtime-gate.json"
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        gate["profiling_root_admissions"][0]["logical_root_id"] = "different-root"
        gate_path.write_text(json.dumps(gate), encoding="utf-8")
        self._refresh_gate_hash()

        with self.assertRaisesRegex(CampaignError, "root admissions differ"):
            run_campaign(self.plan_path, self.output_dir)

    def test_runtime_gate_frozen_ledger_must_match_exact_occurrence_identities(
        self,
    ) -> None:
        self._write_plan()
        gate_path = self.base / "runtime-gate.json"
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        frozen_path = self.base / "frozen-plan.json"
        frozen_plan = json.loads(frozen_path.read_text(encoding="utf-8"))
        frozen_plan["requests_by_arm"]["capture"][0]["tree_occurrence_id"] = (
            "different-tree"
        )
        frozen_path.write_text(json.dumps(frozen_plan), encoding="utf-8")
        gate["provenance"]["frozen_plan"]["sha256"] = hashlib.sha256(
            frozen_path.read_bytes()
        ).hexdigest()
        gate_path.write_text(json.dumps(gate), encoding="utf-8")
        self._refresh_gate_hash()

        with self.assertRaisesRegex(CampaignError, "identities differ"):
            run_campaign(self.plan_path, self.output_dir)

    def test_four_measurements_without_fifth_pair_have_no_paired_interval(self) -> None:
        self._write_plan()
        now = [0.0]

        def monotonic() -> float:
            now[0] += 0.1
            return now[0]

        original_run_pair = _CampaignRunner._run_pair

        def run_pair_then_consume_budget(runner, pair, **kwargs):
            result = original_run_pair(runner, pair, **kwargs)
            if pair["pair_id"] == "pair-04":
                now[0] = 5360.0
            return result

        with patch.object(_CampaignRunner, "_run_pair", run_pair_then_consume_budget):
            result = run_campaign(
                self.plan_path,
                self.output_dir,
                execute=True,
                gpu_probe=lambda: "0, 4\n",
                monotonic=monotonic,
            )

        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(len(result["measurement_overheads"]), 4)
        self.assertIsNone(result["paired_interval"])
        self.assertEqual(
            result["paired_interval_status"], "INCOMPLETE_FIVE_PAIR_EXPANSION"
        )
        pair_receipts = [
            pair for pair in result["pairs"] if pair["pair_id"].startswith("pair-")
        ]
        self.assertEqual(
            len([pair for pair in pair_receipts if pair["status"] == "PASS"]), 4
        )
        self.assertTrue((self.output_dir / "pair-05" / "receipt.json").is_file())
        self.assertEqual(result["teardown"]["status"], "PASS")

    def _assert_pair_mismatch_rejected(self, mismatch: str) -> None:
        plan = self._write_plan()
        plan["measurement_pairs"][0]["commands"]["replay"]["env"][mismatch] = "1"
        self.plan_path.write_text(json.dumps(plan), encoding="utf-8")

        result = run_campaign(
            self.plan_path,
            self.output_dir,
            execute=True,
            gpu_probe=lambda: "0, 4\n",
        )

        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["teardown"]["status"], "PASS")
        self.assertTrue((self.output_dir / "teardown.marker").is_file())
        pair_receipt = json.loads(
            (self.output_dir / "pair-01" / "receipt.json").read_text(encoding="utf-8")
        )
        self.assertEqual(pair_receipt["status"], "FAILED")

    def test_pair_rejects_output_mismatch_and_still_tears_down(self) -> None:
        self._assert_pair_mismatch_rejected("TEST_MISMATCH_OUTPUT")

    def test_pair_rejects_cache_mismatch_and_still_tears_down(self) -> None:
        self._assert_pair_mismatch_rejected("TEST_MISMATCH_CACHE")

    def test_budget_block_creates_not_started_receipt_and_runs_reserved_teardown(
        self,
    ) -> None:
        self._write_plan(
            gate_seconds={"capture": 1000, "generated": 1500, "replay": 2500}
        )
        now = [0.0]
        original_run_setup = _CampaignRunner._run_setup

        def run_setup_then_consume_budget(runner):
            result = original_run_setup(runner)
            now[0] = 101.0
            return result

        with patch.object(_CampaignRunner, "_run_setup", run_setup_then_consume_budget):
            result = run_campaign(
                self.plan_path,
                self.output_dir,
                execute=True,
                gpu_probe=lambda: "0, 4\n",
                monotonic=lambda: now[0],
            )

        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertIn("budget", result["reason"])
        self.assertEqual(result["teardown"]["status"], "PASS")
        self.assertEqual(
            json.loads((self.output_dir / "capture" / "receipt.json").read_text())[
                "status"
            ],
            "NOT_STARTED",
        )
        self.assertTrue((self.output_dir / "teardown.marker").is_file())
        self.assertEqual(result["budget"]["runtime_gate_gpu_seconds"], 5000)
        self.assertGreaterEqual(
            result["budget"]["elapsed_seconds_including_gate_and_teardown"], 5101
        )

    def test_invalid_runner_timing_receipt_is_rejected_before_execution(self) -> None:
        self._write_plan()

        with self.assertRaisesRegex(CampaignError, "invalid gpu_active_seconds"):
            receipt = self.base / "generated-receipt.json"
            receipt.write_text(
                json.dumps(
                    {"schema_version": 1, "arm": "generated", "gpu_active_seconds": 0}
                ),
                encoding="utf-8",
            )
            plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
            gate_path = Path(plan["correctness_gate"]["path"])
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            gate["provenance"]["runner_receipts"]["generated"]["sha256"] = (
                hashlib.sha256(receipt.read_bytes()).hexdigest()
            )
            gate_path.write_text(json.dumps(gate), encoding="utf-8")
            plan["correctness_gate"]["sha256"] = hashlib.sha256(
                gate_path.read_bytes()
            ).hexdigest()
            self.plan_path.write_text(json.dumps(plan), encoding="utf-8")
            run_campaign(
                self.plan_path,
                self.output_dir,
                execute=True,
                gpu_probe=lambda: self.fail(
                    "missing gate timing must fail before GPU probe"
                ),
            )

    def test_overflowing_runner_timing_total_is_rejected(self) -> None:
        self._write_plan(
            gate_seconds={"capture": 1e308, "generated": 1e308, "replay": 1e308}
        )

        with self.assertRaisesRegex(CampaignError, "invalid total gpu_active_seconds"):
            run_campaign(self.plan_path, self.output_dir)

    def test_missing_runner_receipt_is_rejected_before_execution(self) -> None:
        self._write_plan()
        receipt = self.base / "capture-receipt.json"
        receipt.unlink()

        with self.assertRaisesRegex(CampaignError, "artifact is missing"):
            run_campaign(
                self.plan_path,
                self.output_dir,
                execute=True,
                gpu_probe=lambda: self.fail(
                    "missing receipt must fail before GPU probe"
                ),
            )

    def test_budget_cap_and_teardown_reserve_cannot_be_weakened(self) -> None:
        for field, value, message in (
            (
                "hard_budget_seconds",
                MAX_BUDGET_SECONDS + 1,
                "hard_budget_seconds cannot exceed",
            ),
            (
                "teardown_reserve_seconds",
                DEFAULT_TEARDOWN_RESERVE_SECONDS - 1,
                "teardown_reserve_seconds must equal",
            ),
        ):
            with self.subTest(field=field):
                self._write_plan()
                plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
                plan[field] = value
                self.plan_path.write_text(json.dumps(plan), encoding="utf-8")

                with self.assertRaisesRegex(CampaignError, message):
                    run_campaign(self.plan_path, self.output_dir)

                self.output_dir = self.base / f"run-{field}"

    def test_gpu_probe_requires_one_physical_gpu_zero_row(self) -> None:
        self.assertEqual(parse_gpu0_memory("0, 4\n1, 23801\n"), 4)
        with self.assertRaises(CampaignError):
            parse_gpu0_memory("1, 4\n")
        with self.assertRaises(CampaignError):
            parse_gpu0_memory("0, 4\n0, 5\n")


if __name__ == "__main__":
    unittest.main()
