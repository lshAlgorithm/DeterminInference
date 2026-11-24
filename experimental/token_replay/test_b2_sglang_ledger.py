from __future__ import annotations

import hashlib
import json
import base64
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import b2_cpu_gate
from b2_sglang_ledger import RuntimeLedgerError, build_event_bundle


def reference_fixture() -> dict:
    rows = [
        {
            "session": f"{index:03}.json",
            "request": 0,
            "output_ids": [index + 1, index + 2],
        }
        for index in range(8)
    ]
    return {"complete": True, "requests_exact": len(rows), "rows": rows}


def _write_hashed_json(base_dir: Path, name: str, value: dict) -> dict[str, str]:
    path = base_dir / name
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _artifact_ref(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _refresh_receipt(manifest: dict, base_dir: Path, arm: str) -> dict:
    entry = manifest["runner_receipts"][arm]
    path = Path(entry["path"])
    if not path.is_absolute():
        path = base_dir / path
    entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_sha256(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _occurrence_header(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def runtime_fixture(base_dir: Path) -> tuple[dict, dict]:
    reference = reference_fixture()
    cohort = b2_cpu_gate.select_cohort(reference)
    requests_by_arm = {}
    warmup_requests_by_arm = {}
    request_logs = {}
    runner_receipts = {}
    request_log_partitions = {}
    aiperf_artifacts = {}
    execution_config = _write_hashed_json(
        base_dir,
        "execution-config.json",
        {
            "schema_version": 1,
            "arms": {
                arm: {
                    "argv": ["fixture-runner"],
                    "cwd": str(base_dir.resolve()),
                    "timeout_seconds": 30,
                    "env": {},
                    "aiperf_artifact_dir": f"aiperf-{arm}",
                    "gpu_lifecycle_included": True,
                }
                for arm in b2_cpu_gate.ARMS
            },
        },
    )

    for arm in b2_cpu_gate.ARMS:
        requests = []
        warmup_requests = []
        log_lines = []
        for index, session in enumerate(cohort["selected_sessions"]):
            source_index = int(session[:3])
            profile_occurrence = {
                "source_file": session,
                "source_row": 0,
                "phase": "profiling",
                "logical_root_id": f"root-{index}",
                "tree_occurrence_id": f"tree-{index}",
                "turn": 0,
                "reset_ordinal": 0,
                "cache_salt_class": f"salt-{index}",
            }
            profile_id = f"{arm}-request-{index}-profiling"
            profile_request = {"request_id": profile_id, **profile_occurrence}
            requests.append(profile_request)
            server_request_id = f"server-{arm}-{index}-profiling"
            log_lines.append(
                json.dumps(
                    {
                        "event": "request.finished",
                        "rid": server_request_id,
                        "headers": {
                            "x-request-id": profile_request["request_id"],
                            "x-b2-occurrence": _occurrence_header(profile_occurrence),
                        },
                        "out": {
                            "output_ids": [source_index + 1, source_index + 2],
                            "meta_info": {
                                "id": server_request_id,
                                "prompt_tokens": 16,
                                "cached_tokens": 7,
                                "completion_tokens": 2,
                                "total_retractions": 0,
                            },
                        },
                    }
                )
            )
        for index, session in enumerate(cohort["selected_sessions"][:2]):
            source_index = int(session[:3])
            warmup_occurrence = {
                "source_file": session,
                "source_row": 0,
                "phase": "warmup",
                "logical_root_id": f"warmup-root-{index}",
                "tree_occurrence_id": f"warmup-tree-{index}",
                "turn": index,
                "reset_ordinal": 0,
                "cache_salt_class": f"warmup-salt-{index}",
            }
            warmup_request = {
                "request_id": f"{arm}-warmup-{index}-warmup",
                **warmup_occurrence,
                "locator": {"outer_index": 0, "inner_index": None},
            }
            warmup_requests.append(warmup_request)
            server_request_id = f"server-{arm}-{index}-warmup"
            log_lines.append(
                json.dumps(
                    {
                        "event": "request.finished",
                        "rid": server_request_id,
                        "headers": {
                            "x-request-id": warmup_request["request_id"],
                            "x-b2-occurrence": _occurrence_header(warmup_occurrence),
                        },
                        "out": {
                            "output_ids": [source_index + 1, source_index + 2],
                            "meta_info": {
                                "id": server_request_id,
                                "prompt_tokens": 16,
                                "cached_tokens": 7,
                                "completion_tokens": 2,
                                "total_retractions": 0,
                            },
                        },
                    }
                )
            )
        requests_by_arm[arm] = requests
        warmup_requests_by_arm[arm] = warmup_requests
        log_path = base_dir / f"{arm}.log"
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        control_path = base_dir / f"{arm}-control.log"
        control_path.write_text(
            json.dumps(
                {
                    "event": "request.finished",
                    "rid": "HEALTH_CHECK_fixture",
                    "headers": {},
                }
            )
            + "\n"
        )
        raw_path = base_dir / f"{arm}-raw.log"
        raw_path.write_bytes(log_path.read_bytes() + control_path.read_bytes())
        request_log_partitions[arm] = {
            "raw": _artifact_ref(raw_path),
            "business": _artifact_ref(log_path),
            "control_sidecar": _artifact_ref(control_path),
            "business_request_finished_count": len(requests) + len(warmup_requests),
            "control_request_finished_count": 1,
        }
        request_logs[arm] = [_artifact_ref(log_path)]

        artifact_dir = base_dir / f"aiperf-{arm}"
        artifact_dir.mkdir()
        profile_ids = [item["request_id"] for item in requests]
        warmup_ids = [item["request_id"] for item in warmup_requests]
        raw_rows = []
        for phase, phase_ids in (("profiling", profile_ids), ("warmup", warmup_ids)):
            for index, request_id in enumerate(phase_ids):
                raw_rows.append(
                    {
                        "metadata": {
                            "x_request_id": request_id,
                            "benchmark_phase": phase,
                            "context_overflow_skip": False,
                            "was_cancelled": False,
                            "request_start_ns": index + 1
                            if phase == "profiling"
                            else 0,
                        },
                        "error": None,
                    }
                )
        (artifact_dir / "profile_export.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in raw_rows)
        )
        (artifact_dir / "outputs.json").write_text(
            json.dumps(
                {
                    "data": [
                        {"x_request_id": value, "benchmark_phase": "profiling"}
                        for value in profile_ids
                    ],
                    "warmup": [
                        {"x_request_id": value, "benchmark_phase": "warmup"}
                        for value in warmup_ids
                    ],
                }
            )
        )
        phases = []
        client_log = base_dir / f"{arm}-aiperf.log"
        client_lines = []
        for phase, count in (
            ("warmup", len(warmup_ids)),
            ("profiling", len(profile_ids)),
        ):
            metrics_name = f"{phase}-metrics.json"
            (artifact_dir / metrics_name).write_text(
                json.dumps({"benchmark_id": f"{arm}-{phase}"})
            )
            phases.append({"phase_kind": phase, "metrics_json": metrics_name})
            client_lines.extend(
                [
                    f"NOTICE Phase {phase} ({phase}) sending complete | sent={count}, completed={count}, in_flight=0",
                    f"NOTICE Phase {phase} ({phase}) complete | completed={count}, cancelled=0, errors=0",
                    f"INFO PhaseRecordsStats(phase_kind='{phase}', final_requests_completed={count}, final_requests_cancelled=0, final_request_errors=0)",
                ]
            )
        (artifact_dir / "phase_manifest.json").write_text(
            json.dumps({"phases": phases})
        )
        client_log.write_text("\n".join(client_lines) + "\n")
        aiperf_artifacts[arm] = {
            "phase_manifest": _artifact_ref(artifact_dir / "phase_manifest.json"),
            "outputs": _artifact_ref(artifact_dir / "outputs.json"),
            "per_request_export": _artifact_ref(artifact_dir / "profile_export.jsonl"),
            "client_log": _artifact_ref(client_log),
            "metrics": {
                phase: _artifact_ref(artifact_dir / f"{phase}-metrics.json")
                for phase in ("warmup", "profiling")
            },
        }
    frozen_plan = _write_hashed_json(
        base_dir,
        "frozen-plan.json",
        {
            "schema_version": 1,
            "reference_sha256": _canonical_sha256(reference),
            "selected_sessions": cohort["selected_sessions"],
            "profiling_root_admissions": [
                {
                    "source_file": cohort["selected_sessions"][index],
                    "logical_root_id": f"root-{index}",
                }
                for index in range(4)
            ],
            "requests_by_arm": requests_by_arm,
            "warmup_requests_by_arm": warmup_requests_by_arm,
        },
    )
    for arm in b2_cpu_gate.ARMS:
        requests = requests_by_arm[arm]
        warmup_requests = warmup_requests_by_arm[arm]
        runner_receipts[arm] = _write_hashed_json(
            base_dir,
            f"{arm}-receipt.json",
            {
                "schema_version": 1,
                "run_id": f"run-{arm}",
                "arm": arm,
                "frozen_plan_sha256": frozen_plan["sha256"],
                "execution_config_sha256": execution_config["sha256"],
                "loaded_sessions": cohort["selected_sessions"],
                "profiling_root_admissions": [
                    {
                        "source_file": cohort["selected_sessions"][index],
                        "logical_root_id": f"root-{index}",
                    }
                    for index in range(4)
                ],
                "request_ids": [item["request_id"] for item in requests],
                "warmup_request_ids": [item["request_id"] for item in warmup_requests],
                "sent": 6,
                "completed": 6,
                "errors": 0,
                "skips": 0,
                "cancellations": 0,
                "pending": 0,
                "phase_drains": {
                    "warmup": {
                        "sent": len(warmup_requests),
                        "completed": len(warmup_requests),
                        "errors": 0,
                        "cancellations": 0,
                        "pending": 0,
                    },
                    "profiling": {
                        "sent": len(requests),
                        "completed": len(requests),
                        "errors": 0,
                        "cancellations": 0,
                        "pending": 0,
                    },
                },
                "skip_evidence": {
                    "count": 0,
                    "per_request_artifact": aiperf_artifacts[arm]["per_request_export"],
                },
                "aiperf_artifacts": aiperf_artifacts[arm],
                "request_log_partition": request_log_partitions[arm],
                "gpu_active_seconds": 12.5,
                "arm_timing": {
                    "wall_start_utc": "2026-11-20T00:00:00Z",
                    "wall_end_utc": "2026-11-20T00:00:12.500000Z",
                    "monotonic_start_seconds": 100.0,
                    "monotonic_end_seconds": 112.5,
                    "elapsed_seconds": 12.5,
                    "gpu_active_seconds": 12.5,
                    "clock_semantics": "fixture monotonic interval",
                },
            },
        )

    manifest = {
        "schema_version": 1,
        "server_logging": {
            "enabled": True,
            "format": "json",
            "level": 3,
            "incremental_streaming_output": False,
            "request_id_header_logged": True,
            "log_exceeded_ms": -1,
            "dedicated_target": True,
        },
        "frozen_plan": frozen_plan,
        "request_logs": request_logs,
        "runner_receipts": runner_receipts,
        "request_log_partitions": request_log_partitions,
        "aiperf_artifacts": aiperf_artifacts,
        "execution_config": execution_config,
    }
    return reference, manifest


def _rewrite_log(manifest: dict, base_dir: Path, arm: str, text: str) -> None:
    entry = manifest["request_logs"][arm][0]
    path = base_dir / entry["path"]
    path.write_text(text, encoding="utf-8")
    entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    partition = manifest["request_log_partitions"][arm]
    business_path = Path(partition["business"]["path"])
    if not business_path.is_absolute():
        business_path = base_dir / business_path
    raw_path = Path(partition["raw"]["path"])
    if not raw_path.is_absolute():
        raw_path = base_dir / raw_path
    control_path = Path(partition["control_sidecar"]["path"])
    if not control_path.is_absolute():
        control_path = base_dir / control_path
    raw_path.write_bytes(business_path.read_bytes() + control_path.read_bytes())
    partition["business"] = _artifact_ref(business_path)
    partition["raw"] = _artifact_ref(raw_path)
    partition["business_request_finished_count"] = sum(
        1
        for line in business_path.read_text().splitlines()
        if json.loads(line).get("event") == "request.finished"
    )
    receipt_entry = manifest["runner_receipts"][arm]
    receipt_path = Path(receipt_entry["path"])
    if not receipt_path.is_absolute():
        receipt_path = base_dir / receipt_path
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["request_log_partition"] = {
        **partition,
        "raw": _artifact_ref(raw_path),
        "business": _artifact_ref(business_path),
        "control_sidecar": _artifact_ref(control_path),
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_entry["sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()


class TestB2SglangLedger(unittest.TestCase):
    def test_native_request_finished_shape_builds_symmetric_gate_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            base_dir = Path(temporary_dir)
            reference, manifest = runtime_fixture(base_dir)
            bundle = build_event_bundle(manifest, base_dir, reference)
            result = b2_cpu_gate.evaluate_gate(reference, bundle, fixture_only=True)

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["event_counts"], {arm: 4 for arm in b2_cpu_gate.ARMS})
        self.assertEqual(len(bundle["warmup_events"]), 6)
        self.assertTrue(
            all(event["phase"] == "warmup" for event in bundle["warmup_events"])
        )
        self.assertEqual(bundle["events"][0]["output_ids"], [2, 3])
        self.assertEqual(bundle["events"][0]["cache_hit_tokens"], 7)
        self.assertEqual(bundle["events"][0]["new_prefill_tokens"], 9)
        self.assertEqual(bundle["events"][0]["total_retractions"], 0)
        self.assertEqual(
            bundle["provenance"]["frozen_plan"]["sha256"],
            manifest["frozen_plan"]["sha256"],
        )

    def test_runner_receipt_requires_exactly_four_distinct_root_admissions(
        self,
    ) -> None:
        for count in (3, 5):
            with (
                self.subTest(count=count),
                tempfile.TemporaryDirectory() as temporary_dir,
            ):
                base_dir = Path(temporary_dir)
                reference, manifest = runtime_fixture(base_dir)
                receipt_entry = manifest["runner_receipts"]["capture"]
                receipt_path = base_dir / receipt_entry["path"]
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                admissions = receipt["profiling_root_admissions"]
                receipt["profiling_root_admissions"] = admissions[:count]
                if count == 5:
                    receipt["profiling_root_admissions"].append(
                        {
                            "source_file": receipt["loaded_sessions"][0],
                            "logical_root_id": "root-extra",
                        }
                    )
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                receipt_entry["sha256"] = hashlib.sha256(
                    receipt_path.read_bytes()
                ).hexdigest()
                with self.assertRaisesRegex(RuntimeLedgerError, "exactly four roots"):
                    build_event_bundle(manifest, base_dir, reference)

    def test_duplicate_root_admission_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            base_dir = Path(temporary_dir)
            reference, manifest = runtime_fixture(base_dir)
            receipt_entry = manifest["runner_receipts"]["capture"]
            receipt_path = base_dir / receipt_entry["path"]
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["profiling_root_admissions"][1] = dict(
                receipt["profiling_root_admissions"][0]
            )
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            receipt_entry["sha256"] = hashlib.sha256(
                receipt_path.read_bytes()
            ).hexdigest()
            with self.assertRaisesRegex(RuntimeLedgerError, "four distinct roots"):
                build_event_bundle(manifest, base_dir, reference)

    def test_warmup_occurrence_mappings_must_match_across_arms(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            base_dir = Path(temporary_dir)
            reference, manifest = runtime_fixture(base_dir)
            plan_entry = manifest["frozen_plan"]
            plan_path = base_dir / plan_entry["path"]
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["warmup_requests_by_arm"]["generated"][0]["logical_root_id"] = (
                "other-root"
            )
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            plan_entry["sha256"] = hashlib.sha256(plan_path.read_bytes()).hexdigest()

            with self.assertRaisesRegex(
                RuntimeLedgerError, "warmup occurrence mappings are asymmetric"
            ):
                build_event_bundle(manifest, base_dir, reference)

    def test_runner_warmup_request_ids_must_match_the_frozen_list(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            base_dir = Path(temporary_dir)
            reference, manifest = runtime_fixture(base_dir)
            receipt_entry = manifest["runner_receipts"]["capture"]
            receipt_path = base_dir / receipt_entry["path"]
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["warmup_request_ids"][0] = "unexpected-warmup"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            receipt_entry["sha256"] = hashlib.sha256(
                receipt_path.read_bytes()
            ).hexdigest()

            with self.assertRaisesRegex(
                RuntimeLedgerError,
                "warmup_request_ids differ from the frozen warmup plan",
            ):
                build_event_bundle(manifest, base_dir, reference)

    def test_runner_receipt_requires_a_clean_full_drain(self) -> None:
        for field, value in (
            ("completed", 7),
            ("errors", 1),
            ("skips", 1),
            ("cancellations", 1),
            ("pending", 1),
        ):
            with (
                self.subTest(field=field),
                tempfile.TemporaryDirectory() as temporary_dir,
            ):
                base_dir = Path(temporary_dir)
                reference, manifest = runtime_fixture(base_dir)
                receipt_entry = manifest["runner_receipts"]["capture"]
                receipt_path = base_dir / receipt_entry["path"]
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                receipt[field] = value
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                receipt_entry["sha256"] = hashlib.sha256(
                    receipt_path.read_bytes()
                ).hexdigest()
                with self.assertRaisesRegex(
                    RuntimeLedgerError, "did not drain cleanly"
                ):
                    build_event_bundle(manifest, base_dir, reference)

    def test_runner_receipt_requires_positive_hash_bound_timing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            base_dir = Path(temporary_dir)
            reference, manifest = runtime_fixture(base_dir)
            receipt_entry = manifest["runner_receipts"]["capture"]
            receipt_path = Path(receipt_entry["path"])
            if not receipt_path.is_absolute():
                receipt_path = base_dir / receipt_path
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["gpu_active_seconds"] = 0
            receipt["arm_timing"]["gpu_active_seconds"] = 0
            receipt["arm_timing"]["elapsed_seconds"] = 0
            receipt["arm_timing"]["monotonic_end_seconds"] = receipt["arm_timing"][
                "monotonic_start_seconds"
            ]
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            receipt_entry["sha256"] = hashlib.sha256(
                receipt_path.read_bytes()
            ).hexdigest()

            with self.assertRaisesRegex(RuntimeLedgerError, "monotonic GPU timing"):
                build_event_bundle(manifest, base_dir, reference)

    def test_aiperf_skip_evidence_is_recomputed_from_hashed_raw_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            base_dir = Path(temporary_dir)
            reference, manifest = runtime_fixture(base_dir)
            receipt_entry = manifest["runner_receipts"]["capture"]
            receipt_path = Path(receipt_entry["path"])
            if not receipt_path.is_absolute():
                receipt_path = base_dir / receipt_path
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            raw_path = Path(receipt["aiperf_artifacts"]["per_request_export"]["path"])
            rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
            rows[0]["metadata"]["context_overflow_skip"] = True
            raw_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            raw_ref = _artifact_ref(raw_path)
            receipt["aiperf_artifacts"]["per_request_export"] = raw_ref
            receipt["skip_evidence"]["per_request_artifact"] = raw_ref
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            receipt_entry["sha256"] = hashlib.sha256(
                receipt_path.read_bytes()
            ).hexdigest()

            with self.assertRaisesRegex(
                RuntimeLedgerError, "AIPerf artifacts failed independent validation"
            ):
                build_event_bundle(manifest, base_dir, reference)

    def test_aiperf_first_seen_root_order_must_match_frozen_admissions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            base_dir = Path(temporary_dir)
            reference, manifest = runtime_fixture(base_dir)
            receipt_entry = manifest["runner_receipts"]["capture"]
            receipt_path = Path(receipt_entry["path"])
            if not receipt_path.is_absolute():
                receipt_path = base_dir / receipt_path
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            raw_path = Path(receipt["aiperf_artifacts"]["per_request_export"]["path"])
            rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
            profile_rows = [
                row for row in rows if row["metadata"]["benchmark_phase"] == "profiling"
            ]
            profile_rows[0]["metadata"]["request_start_ns"] = 100
            profile_rows[1]["metadata"]["request_start_ns"] = 1
            raw_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            raw_ref = _artifact_ref(raw_path)
            receipt["aiperf_artifacts"]["per_request_export"] = raw_ref
            receipt["skip_evidence"]["per_request_artifact"] = raw_ref
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            receipt_entry["sha256"] = hashlib.sha256(
                receipt_path.read_bytes()
            ).hexdigest()

            with self.assertRaisesRegex(
                RuntimeLedgerError, "admissions differ from AIPerf first-seen roots"
            ):
                build_event_bundle(manifest, base_dir, reference)

    def test_warmup_request_finished_ledger_requires_each_frozen_occurrence_once(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            base_dir = Path(temporary_dir)
            reference, manifest = runtime_fixture(base_dir)
            path = base_dir / "capture.log"
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            _rewrite_log(
                manifest,
                base_dir,
                "capture",
                "".join(
                    json.dumps(record) + "\n"
                    for record in records
                    if record["headers"]["x-request-id"].endswith("-profiling")
                ),
            )
            with self.assertRaisesRegex(
                RuntimeLedgerError, "warmup request.finished ledger is missing"
            ):
                build_event_bundle(manifest, base_dir, reference)

        with tempfile.TemporaryDirectory() as temporary_dir:
            base_dir = Path(temporary_dir)
            reference, manifest = runtime_fixture(base_dir)
            path = base_dir / "capture.log"
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            warmup = next(
                record
                for record in records
                if record["headers"]["x-request-id"].endswith("-warmup")
            )
            _rewrite_log(
                manifest,
                base_dir,
                "capture",
                path.read_text(encoding="utf-8") + json.dumps(warmup) + "\n",
            )
            with self.assertRaisesRegex(
                RuntimeLedgerError, "warmup request.finished ledger duplicates"
            ):
                build_event_bundle(manifest, base_dir, reference)

    def test_warmup_request_finished_phase_must_match_request_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            base_dir = Path(temporary_dir)
            reference, manifest = runtime_fixture(base_dir)
            path = base_dir / "capture.log"
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            warmup = next(
                record
                for record in records
                if record["headers"]["x-request-id"].endswith("-warmup")
            )
            occurrence = json.loads(
                base64.urlsafe_b64decode(
                    warmup["headers"]["x-b2-occurrence"]
                    + "=" * (-len(warmup["headers"]["x-b2-occurrence"]) % 4)
                )
            )
            occurrence["phase"] = "profiling"
            warmup["headers"]["x-b2-occurrence"] = _occurrence_header(occurrence)
            _rewrite_log(
                manifest,
                base_dir,
                "capture",
                "".join(json.dumps(record) + "\n" for record in records),
            )
            with self.assertRaisesRegex(
                RuntimeLedgerError, "differs from the frozen mapping"
            ):
                build_event_bundle(manifest, base_dir, reference)

    def test_level_two_or_incremental_logging_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            base_dir = Path(temporary_dir)
            reference, manifest = runtime_fixture(base_dir)
            manifest["server_logging"]["level"] = 2
            with self.assertRaisesRegex(RuntimeLedgerError, "level"):
                build_event_bundle(manifest, base_dir, reference)

            manifest["server_logging"]["level"] = 3
            manifest["server_logging"]["incremental_streaming_output"] = True
            with self.assertRaisesRegex(
                RuntimeLedgerError, "incremental_streaming_output"
            ):
                build_event_bundle(manifest, base_dir, reference)

    def test_missing_request_id_header_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            base_dir = Path(temporary_dir)
            reference, manifest = runtime_fixture(base_dir)
            path = base_dir / "capture.log"
            record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            record["headers"] = {}
            _rewrite_log(manifest, base_dir, "capture", json.dumps(record) + "\n")
            with self.assertRaisesRegex(RuntimeLedgerError, "x-request-id"):
                build_event_bundle(manifest, base_dir, reference)

    def test_server_occurrence_header_is_required_and_bound_to_frozen_plan(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            base_dir = Path(temporary_dir)
            reference, manifest = runtime_fixture(base_dir)
            path = base_dir / "capture.log"
            record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            del record["headers"]["x-b2-occurrence"]
            _rewrite_log(manifest, base_dir, "capture", json.dumps(record) + "\n")
            with self.assertRaisesRegex(RuntimeLedgerError, "x-b2-occurrence"):
                build_event_bundle(manifest, base_dir, reference)

            with tempfile.TemporaryDirectory(dir=base_dir) as nested_dir:
                reference, manifest = runtime_fixture(Path(nested_dir))
                path = Path(nested_dir) / "capture.log"
                record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
                padded = record["headers"]["x-b2-occurrence"] + "=" * (
                    -len(record["headers"]["x-b2-occurrence"]) % 4
                )
                occurrence = json.loads(base64.urlsafe_b64decode(padded))
                occurrence["tree_occurrence_id"] = "forged-tree"
                record["headers"]["x-b2-occurrence"] = _occurrence_header(occurrence)
                _rewrite_log(
                    manifest, Path(nested_dir), "capture", json.dumps(record) + "\n"
                )
                with self.assertRaisesRegex(
                    RuntimeLedgerError, "differs from the frozen mapping"
                ):
                    build_event_bundle(manifest, Path(nested_dir), reference)

    def test_truncated_token_vector_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            base_dir = Path(temporary_dir)
            reference, manifest = runtime_fixture(base_dir)
            path = base_dir / "capture.log"
            record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            record["out"]["output_ids"] = [1, "...", 3]
            _rewrite_log(manifest, base_dir, "capture", json.dumps(record) + "\n")
            with self.assertRaisesRegex(RuntimeLedgerError, "truncated"):
                build_event_bundle(manifest, base_dir, reference)

    def test_cache_counts_outside_prompt_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            base_dir = Path(temporary_dir)
            reference, manifest = runtime_fixture(base_dir)
            path = base_dir / "capture.log"
            record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            record["out"]["meta_info"]["cached_tokens"] = 17
            _rewrite_log(manifest, base_dir, "capture", json.dumps(record) + "\n")
            with self.assertRaisesRegex(RuntimeLedgerError, "exceeds prompt_tokens"):
                build_event_bundle(manifest, base_dir, reference)

    def test_missing_finished_event_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            base_dir = Path(temporary_dir)
            reference, manifest = runtime_fixture(base_dir)
            _rewrite_log(manifest, base_dir, "replay", "")
            with self.assertRaisesRegex(RuntimeLedgerError, "no request.finished"):
                build_event_bundle(manifest, base_dir, reference)

    def test_modified_request_log_is_rejected_by_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            base_dir = Path(temporary_dir)
            reference, manifest = runtime_fixture(base_dir)
            with (base_dir / "capture.log").open("a", encoding="utf-8") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(RuntimeLedgerError, "SHA-256 mismatch"):
                build_event_bundle(manifest, base_dir, reference)

    def test_runner_receipt_is_bound_to_frozen_plan_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            base_dir = Path(temporary_dir)
            reference, manifest = runtime_fixture(base_dir)
            receipt_entry = manifest["runner_receipts"]["capture"]
            receipt_path = base_dir / receipt_entry["path"]
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["frozen_plan_sha256"] = "0" * 64
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            receipt_entry["sha256"] = hashlib.sha256(
                receipt_path.read_bytes()
            ).hexdigest()
            with self.assertRaisesRegex(
                RuntimeLedgerError, "does not reference the frozen plan"
            ):
                build_event_bundle(manifest, base_dir, reference)

    def test_modified_runner_receipt_is_rejected_by_its_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            base_dir = Path(temporary_dir)
            reference, manifest = runtime_fixture(base_dir)
            receipt_entry = manifest["runner_receipts"]["capture"]
            receipt_path = base_dir / receipt_entry["path"]
            with receipt_path.open("a", encoding="utf-8") as stream:
                stream.write(" ")
            with self.assertRaisesRegex(
                RuntimeLedgerError, "runner_receipts.capture SHA-256 mismatch"
            ):
                build_event_bundle(manifest, base_dir, reference)

    def test_inline_occurrences_and_drain_do_not_replace_independent_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            base_dir = Path(temporary_dir)
            reference, manifest = runtime_fixture(base_dir)
            manifest["requests_by_arm"] = {arm: [] for arm in b2_cpu_gate.ARMS}
            manifest["drain"] = {"by_arm": {arm: {} for arm in b2_cpu_gate.ARMS}}
            del manifest["frozen_plan"]
            del manifest["runner_receipts"]
            with self.assertRaisesRegex(RuntimeLedgerError, "frozen_plan"):
                build_event_bundle(manifest, base_dir, reference)

    def test_frozen_plan_reference_hash_must_match_gate_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            base_dir = Path(temporary_dir)
            reference, manifest = runtime_fixture(base_dir)
            plan_entry = manifest["frozen_plan"]
            plan_path = base_dir / plan_entry["path"]
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["reference_sha256"] = "0" * 64
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            plan_entry["sha256"] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(
                RuntimeLedgerError, "does not match the gate reference"
            ):
                build_event_bundle(manifest, base_dir, reference)

    def test_runner_request_ids_must_match_frozen_mapping_and_server_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            base_dir = Path(temporary_dir)
            reference, manifest = runtime_fixture(base_dir)
            receipt_entry = manifest["runner_receipts"]["capture"]
            receipt_path = base_dir / receipt_entry["path"]
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["request_ids"].pop()
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            receipt_entry["sha256"] = hashlib.sha256(
                receipt_path.read_bytes()
            ).hexdigest()
            with self.assertRaisesRegex(
                RuntimeLedgerError, "request_ids differ from the frozen plan"
            ):
                build_event_bundle(manifest, base_dir, reference)

    def test_runtime_cli_builds_only_from_hash_verified_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            base_dir = Path(temporary_dir)
            reference, manifest = runtime_fixture(base_dir)
            reference_path = base_dir / "reference.json"
            manifest_path = base_dir / "runtime-manifest.json"
            output_path = base_dir / "gate-result.json"
            reference_path.write_text(json.dumps(reference), encoding="utf-8")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(b2_cpu_gate.__file__)),
                    "--reference",
                    str(reference_path),
                    "--runtime-manifest",
                    str(manifest_path),
                    "--output",
                    str(output_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["runtime_artifacts_verified"])
        self.assertEqual(report["evidence_scope"], "runtime")
        self.assertEqual(
            set(report["provenance"]["runner_receipts"]), set(b2_cpu_gate.ARMS)
        )


if __name__ == "__main__":
    unittest.main()
