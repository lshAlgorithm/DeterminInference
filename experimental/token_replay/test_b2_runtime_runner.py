from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

import b2_cpu_gate
from b2_runtime_runner import ARMS, RuntimeRunnerError, run_b2_plan


FAKE_RUNNER = """
import base64
import json
import os
from pathlib import Path
import sys

arm = os.environ["AGENTX_B2_ARM"]
expected_mode = "recorded" if arm == "replay" else "generated"
assert os.environ["AGENTX_REPLAY_MODE"] == expected_mode
assert os.environ["AGENTX_TOKEN_TRACE"] == os.environ["B2_DATASET_DIR"]
assert os.environ["AGENTX_NUM_SESSIONS"] == "4"
assert os.environ["SGLANG_LOG_REQUEST_HEADERS"] == "x-request-id,x-b2-occurrence"
request_ids = json.loads(os.environ["B2_EXPECTED_REQUEST_IDS_JSON"])
warmup_request_ids = json.loads(os.environ["B2_EXPECTED_WARMUP_REQUEST_IDS_JSON"])
warmup_occurrences = json.loads(os.environ["B2_EXPECTED_WARMUP_OCCURRENCES_JSON"])
sessions = json.loads(os.environ["B2_SELECTED_SESSIONS_JSON"])
roots = json.loads(os.environ["B2_ROOT_ADMISSIONS_JSON"])
events = []

def event_for(request, occurrence, request_id, suffix):
    server_id = f"server-{arm}-{suffix}"
    encoded = base64.urlsafe_b64encode(
        json.dumps(occurrence, sort_keys=True, separators=(",", ":")).encode()
    ).decode().rstrip("=")
    return {
        "event": "request.finished",
        "rid": server_id,
        "headers": {
            "x-request-id": request_id,
            "x-b2-occurrence": encoded,
        },
        "out": {
            "output_ids": request["output_ids"],
            "meta_info": {
                "id": server_id,
                "prompt_tokens": 16,
                "cached_tokens": 7,
                "completion_tokens": len(request["output_ids"]),
                "total_retractions": 0,
            },
        },
    }

for index, session in enumerate(sessions):
    trace = json.loads((Path(os.environ["B2_DATASET_DIR"]) / session).read_text())
    request = trace["requests"][0]
    assert request["request_id"] == request_ids[index].removesuffix("-profiling")
    occurrence = request["occurrence_identity"]
    events.append(event_for(request, occurrence, request_ids[index], f"{index}-profiling"))
for index, warmup in enumerate(warmup_occurrences):
    trace = json.loads((Path(os.environ["B2_DATASET_DIR"]) / warmup["source_file"]).read_text())
    request = trace["requests"][warmup["locator"]["outer_index"]]
    occurrence = {key: warmup[key] for key in (
        "source_file", "source_row", "phase", "logical_root_id", "tree_occurrence_id",
        "turn", "reset_ordinal", "cache_salt_class",
    )}
    events.append(event_for(request, occurrence, warmup["request_id"], f"warmup-{index}"))
Path(os.environ["B2_REQUEST_LOG"]).write_text(
    "".join(json.dumps(event) + "\\n" for event in events)
    + json.dumps({"event": "request.finished", "rid": "HEALTH_CHECK_probe", "headers": {}}) + "\\n"
)
artifact_dir = Path(os.environ["B2_AIPERF_ARTIFACT_DIR"])
profile_rows = [
    {"metadata": {"x_request_id": request_id, "benchmark_phase": "profiling", "context_overflow_skip": False, "was_cancelled": False, "request_start_ns": index + 1}, "error": None}
    for index, request_id in enumerate(request_ids)
]
warmup_rows = [
    {"metadata": {"x_request_id": request_id, "benchmark_phase": "warmup", "context_overflow_skip": False, "was_cancelled": False, "request_start_ns": 0}, "error": None}
    for request_id in warmup_request_ids
]
(artifact_dir / "profile_export.jsonl").write_text(
    "".join(json.dumps(row) + "\\n" for row in profile_rows + warmup_rows)
)
(artifact_dir / "outputs.json").write_text(json.dumps({
    "data": [{"x_request_id": request_id, "benchmark_phase": "profiling"} for request_id in request_ids],
    "warmup": [{"x_request_id": request_id, "benchmark_phase": "warmup"} for request_id in warmup_request_ids],
}))
phases = []
for phase, count in (("warmup", len(warmup_request_ids)), ("profiling", len(request_ids))):
    metrics_name = f"{phase}-metrics.json"
    (artifact_dir / metrics_name).write_text(json.dumps({"benchmark_id": f"run-{arm}-{phase}"}))
    phases.append({"phase_kind": phase, "metrics_json": metrics_name, "request_drain": {"sent": count, "completed": count, "errors": 0, "cancellations": 0, "pending": 0}})
(artifact_dir / "phase_manifest.json").write_text(json.dumps({"phases": phases}))
for phase, count in (("warmup", len(warmup_request_ids)), ("profiling", len(request_ids))):
    print(f"NOTICE Phase {phase} ({phase}) sending complete | sent={count}, completed={count}, in_flight=0")
    print(f"NOTICE Phase {phase} ({phase}) complete | completed={count}, cancelled=0, errors=0")
    print(f"INFO PhaseRecordsStats(phase_kind='{phase}', final_requests_completed={count}, final_requests_cancelled=0, final_request_errors=0)")
Path(os.environ["B2_OBSERVATION"]).write_text(json.dumps({
    "schema_version": 1,
    "arm": arm,
    "run_id": f"run-{arm}",
    "loaded_sessions": sessions,
    "profiling_root_admissions": roots,
    "request_ids": request_ids,
    "warmup_request_ids": warmup_request_ids,
    "sent": len(request_ids) + len(warmup_request_ids),
    "completed": len(events),
    "errors": 0,
    "skips": 0,
    "cancellations": 0,
    "pending": 0,
    "cwd": os.getcwd(),
}))
"""


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture_plan(base_dir: Path) -> tuple[Path, dict]:
    source_dir = base_dir / "source"
    source_dir.mkdir()
    rows = []
    for index in range(4):
        session = f"{index:03}.json"
        output_ids = [index + 1, index + 2]
        rows.append({"session": session, "request": 0, "output_ids": output_ids})
        _write_json(
            source_dir / session,
            {
                "requests": [
                    {"type": "n", "input_ids": [100 + index], "output_ids": output_ids}
                ]
            },
        )
    reference = {"complete": True, "requests_exact": len(rows), "rows": rows}
    reference_path = base_dir / "reference.json"
    _write_json(reference_path, reference)
    cohort = b2_cpu_gate.select_cohort(reference)
    occurrences = []
    warmup_occurrences = []
    roots = []
    for index, session in enumerate(cohort["selected_sessions"]):
        root_id = f"root-{index}"
        occurrences.append(
            {
                "source_file": session,
                "source_row": 0,
                "phase": "profiling",
                "logical_root_id": root_id,
                "tree_occurrence_id": f"tree-{index}",
                "turn": 0,
                "reset_ordinal": 0,
                "cache_salt_class": f"salt-{index}",
                "locator": {"outer_index": 0, "inner_index": None},
            }
        )
        roots.append({"source_file": session, "logical_root_id": root_id})
        if index < 2:
            warmup_occurrences.append(
                {
                    "source_file": session,
                    "source_row": 0,
                    "phase": "warmup",
                    "logical_root_id": f"warmup-root-{index}",
                    "tree_occurrence_id": f"warmup-tree-{index}",
                    "turn": index,
                    "reset_ordinal": 0,
                    "cache_salt_class": f"warmup-salt-{index}",
                    "locator": {"outer_index": 0, "inner_index": None},
                }
            )

    runner_dir = base_dir / "runner"
    runner_dir.mkdir()
    (runner_dir / "fake_runner.py").write_text(FAKE_RUNNER, encoding="utf-8")
    plan = {
        "schema_version": 1,
        "reference": reference_path.name,
        "source_sessions_dir": source_dir.name,
        "selected_sessions": cohort["selected_sessions"],
        "occurrences": occurrences,
        "warmup_occurrences": warmup_occurrences,
        "profiling_root_admissions": roots,
        "arms": {
            arm: {
                "argv": [sys.executable, "fake_runner.py"],
                "cwd": runner_dir.name,
                "timeout_seconds": 10,
                "aiperf_artifact_dir": "aiperf",
                "gpu_lifecycle_included": True,
            }
            for arm in ARMS
        },
    }
    plan_path = base_dir / "plan.json"
    _write_json(plan_path, plan)
    return plan_path, reference


class TestB2RuntimeRunner(unittest.TestCase):
    def test_runs_symmetric_arms_from_configured_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            base_dir = Path(temporary_dir)
            plan_path, reference = _fixture_plan(base_dir)
            output_dir = base_dir / "run"

            status = run_b2_plan(plan_path, output_dir)

            self.assertEqual(status["status"], "READY_FOR_CPU_GATE")
            self.assertEqual(status["occurrence_count_per_arm"], 4)
            self.assertEqual(status["warmup_occurrence_count_per_arm"], 2)
            manifest = json.loads(
                (output_dir / "runtime-manifest.json").read_text(encoding="utf-8")
            )
            bundle_path = output_dir / "event-bundle.json"
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            self.assertEqual(len(bundle["events"]), 12)
            self.assertEqual(len(bundle["warmup_events"]), 6)
            self.assertEqual(
                {
                    arm: sum(event["arm"] == arm for event in bundle["events"])
                    for arm in ARMS
                },
                {arm: 4 for arm in ARMS},
            )
            self.assertEqual(
                {
                    arm: sum(event["arm"] == arm for event in bundle["warmup_events"])
                    for arm in ARMS
                },
                {arm: 2 for arm in ARMS},
            )
            self.assertEqual(set(manifest["runner_receipts"]), set(ARMS))
            for arm in ARMS:
                observation = json.loads(
                    (output_dir / arm / "runner-observation.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertIn("per_request_export", observation["aiperf_artifacts"])
                self.assertEqual(observation["skips"], 0)
            execution_config = json.loads(
                (output_dir / "execution-config.json").read_text()
            )
            self.assertEqual(
                execution_config["arms"]["capture"]["cwd"],
                str((base_dir / "runner").resolve()),
            )

            gate = b2_cpu_gate.evaluate_gate(reference, bundle, fixture_only=True)
            self.assertEqual(gate["status"], "PASS")
            self.assertEqual(gate["evidence_scope"], "fixture-only")

    def test_explicit_warmups_are_independent_of_four_profile_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            base_dir = Path(temporary_dir)
            plan_path, _reference = _fixture_plan(base_dir)
            status = run_b2_plan(plan_path, base_dir / "run")
            bundle = json.loads((base_dir / "run" / "event-bundle.json").read_text())

        self.assertEqual(status["warmup_occurrence_count_per_arm"], 2)
        self.assertEqual(len(bundle["profiling_root_admissions"]), 4)
        self.assertEqual(
            {event["logical_root_id"] for event in bundle["warmup_events"]},
            {"warmup-root-0", "warmup-root-1"},
        )

    def test_duplicate_explicit_warmups_are_rejected(self) -> None:
        for mutation in ("duplicate",):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as temporary_dir,
            ):
                base_dir = Path(temporary_dir)
                plan_path, _reference = _fixture_plan(base_dir)
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
                plan["warmup_occurrences"].append(dict(plan["warmup_occurrences"][0]))
                _write_json(plan_path, plan)

                with self.assertRaises(RuntimeRunnerError):
                    run_b2_plan(plan_path, base_dir / "run")

    def test_explicit_empty_warmup_requires_zero_warmup_phase_and_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            base_dir = Path(temporary_dir)
            plan_path, reference = _fixture_plan(base_dir)
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["warmup_occurrences"] = []
            _write_json(plan_path, plan)

            status = run_b2_plan(plan_path, base_dir / "run")
            bundle = json.loads((base_dir / "run" / "event-bundle.json").read_text())

            self.assertEqual(status["warmup_occurrence_count_per_arm"], 0)
            self.assertEqual(bundle["warmup_events"], [])
            for arm in ARMS:
                receipt = json.loads(
                    (base_dir / "run" / arm / "runner-receipt.json").read_text()
                )
                self.assertEqual(receipt["phase_drains"]["warmup"]["sent"], 0)
                self.assertEqual(receipt["phase_drains"]["warmup"]["completed"], 0)
                self.assertEqual(
                    receipt["gpu_active_seconds"],
                    receipt["arm_timing"]["elapsed_seconds"],
                )
            self.assertEqual(
                b2_cpu_gate.evaluate_gate(reference, bundle, fixture_only=True)[
                    "status"
                ],
                "PASS",
            )

    def test_prepare_only_validates_and_stages_without_running_arms(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            base_dir = Path(temporary_dir)
            plan_path, _reference = _fixture_plan(base_dir)
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            marker = base_dir / "arm-ran"
            command = [
                sys.executable,
                "-c",
                "from pathlib import Path; Path("
                + repr(str(marker))
                + ").write_text('ran')",
            ]
            for arm in ARMS:
                plan["arms"][arm]["argv"] = command
            _write_json(plan_path, plan)
            output_dir = base_dir / "prepared"

            status = run_b2_plan(plan_path, output_dir, prepare_only=True)

            self.assertEqual(status["status"], "VALIDATED_NO_ARMS_RUN")
            self.assertEqual(status["evidence_scope"], "plan-validation-only")
            self.assertFalse(status["arms_executed"])
            self.assertTrue((output_dir / "frozen-plan.json").is_file())
            self.assertTrue((output_dir / "capture" / "dataset").is_dir())
            self.assertFalse(marker.exists())
            self.assertFalse((output_dir / "runtime-manifest.json").exists())

    def test_root_admissions_must_be_exactly_four(self) -> None:
        for invalid_count in (3, 5):
            with (
                self.subTest(count=invalid_count),
                tempfile.TemporaryDirectory() as temporary_dir,
            ):
                base_dir = Path(temporary_dir)
                plan_path, _reference = _fixture_plan(base_dir)
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
                roots = plan["profiling_root_admissions"]
                if invalid_count == 3:
                    plan["profiling_root_admissions"] = roots[:3]
                else:
                    plan["profiling_root_admissions"] = roots + [
                        {
                            "source_file": roots[0]["source_file"],
                            "logical_root_id": "extra-root",
                        }
                    ]
                _write_json(plan_path, plan)

                with self.assertRaisesRegex(RuntimeRunnerError, "exactly four roots"):
                    run_b2_plan(plan_path, base_dir / "run")

    def test_arm_env_cannot_override_runner_owned_values(self) -> None:
        for name in (
            "AGENTX_TOKEN_TRACE",
            "AGENTX_NUM_SESSIONS",
            "B2_EXPECTED_WARMUP_REQUEST_IDS_JSON",
            "B2_EXPECTED_WARMUP_OCCURRENCES_JSON",
        ):
            with (
                self.subTest(name=name),
                tempfile.TemporaryDirectory() as temporary_dir,
            ):
                base_dir = Path(temporary_dir)
                plan_path, _reference = _fixture_plan(base_dir)
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
                plan["arms"]["capture"]["env"] = {name: "override"}
                _write_json(plan_path, plan)

                with self.assertRaisesRegex(RuntimeRunnerError, "runner-owned keys"):
                    run_b2_plan(plan_path, base_dir / "run")

    def test_timeout_waits_for_remaining_process_group_members_and_records_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            base_dir = Path(temporary_dir)
            plan_path, _reference = _fixture_plan(base_dir)
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            marker = base_dir / "child-finished"
            child = (
                "import pathlib,signal,sys,time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "time.sleep(0.6); pathlib.Path(sys.argv[1]).write_text('done')"
            )
            parent = (
                "import subprocess,sys,time; "
                "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]]); "
                "time.sleep(30)"
            )
            plan["arms"]["capture"]["argv"] = [
                sys.executable,
                "-c",
                parent,
                child,
                str(marker),
            ]
            plan["arms"]["capture"]["timeout_seconds"] = 0.1
            _write_json(plan_path, plan)
            output_dir = base_dir / "run"
            started = time.monotonic()

            with self.assertRaisesRegex(RuntimeRunnerError, "timed out"):
                run_b2_plan(plan_path, output_dir)

            elapsed = time.monotonic() - started
            self.assertGreater(elapsed, 0.35)
            self.assertTrue(marker.is_file())
            failure = json.loads(
                (output_dir / "failure.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failure["status"], "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
