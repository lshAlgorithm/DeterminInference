from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from b2_aiperf_adapter import (
    AIPerfAdapterError,
    create_observation,
    inspect_aiperf_artifacts,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(
    base: Path, *, warmup_ids: list[str] | None = None
) -> tuple[dict[str, Path], list[str], list[str]]:
    warmup_ids = ["warmup-1"] if warmup_ids is None else warmup_ids
    profile_ids = ["profile-1", "profile-2"]
    artifact_dir = base / "aiperf"
    artifact_dir.mkdir()
    phase_specs = []
    for phase, count in (("warmup", len(warmup_ids)), ("profiling", len(profile_ids))):
        metrics = artifact_dir / f"{phase}-metrics.json"
        _write_json(metrics, {"benchmark_id": None})
        phase_specs.append(
            {
                "phase_kind": phase,
                "metrics_json": metrics.name,
                "request_drain": {
                    "sent": count,
                    "completed": count,
                    "errors": 0,
                    "cancellations": 0,
                    "pending": 0,
                },
            }
        )
    _write_json(artifact_dir / "phase_manifest.json", {"phases": phase_specs})

    rows_by_phase = {"profiling": profile_ids, "warmup": warmup_ids}
    outputs = {
        "data": [
            {"x_request_id": request_id, "benchmark_phase": "profiling"}
            for request_id in profile_ids
        ],
        "warmup": [
            {"x_request_id": request_id, "benchmark_phase": "warmup"}
            for request_id in warmup_ids
        ],
    }
    _write_json(artifact_dir / "outputs.json", outputs)
    with (artifact_dir / "profile_export.jsonl").open("w", encoding="utf-8") as stream:
        for phase, request_ids in rows_by_phase.items():
            for request_id in request_ids:
                stream.write(
                    json.dumps(
                        {
                            "metadata": {
                                "x_request_id": request_id,
                                "benchmark_phase": phase,
                                "context_overflow_skip": False,
                                "was_cancelled": False,
                                "request_start_ns": profile_ids.index(request_id) + 1
                                if phase == "profiling"
                                else 0,
                            },
                            "error": None,
                        }
                    )
                    + "\n"
                )

    log_path = base / "client.log"
    with log_path.open("w", encoding="utf-8") as stream:
        for phase, count in (
            ("warmup", len(warmup_ids)),
            ("profiling", len(profile_ids)),
        ):
            stream.write(
                f"NOTICE Phase {phase} ({phase}) sending complete | sent={count}, completed={count}, in_flight=0\n"
            )
            stream.write(
                f"NOTICE Phase {phase} ({phase}) complete | completed={count}, cancelled=0, errors=0\n"
            )
            stream.write(
                "INFO PhaseRecordsStats(phase_kind='{}', final_requests_sent=None, "
                "final_requests_completed={}, final_requests_cancelled=0, final_request_errors=0)\n".format(
                    phase, count
                )
            )
    server_log = base / "server-source.jsonl"
    server_records = [
        {
            "event": "request.finished",
            "rid": f"sglang-{request_id}",
            "headers": {"x-request-id": request_id, "x-b2-occurrence": "frozen"},
        }
        for request_id in profile_ids + warmup_ids
    ]
    server_records.append(
        {"event": "request.finished", "rid": "HEALTH_CHECK_probe-1", "headers": {}}
    )
    server_log.write_text(
        "".join(json.dumps(row) + "\n" for row in server_records), encoding="utf-8"
    )
    return (
        {
            "artifact_dir": artifact_dir,
            "client_log": log_path,
            "server_log": server_log,
            "raw": base / "server-raw.jsonl",
            "business": base / "server-business.jsonl",
            "control": base / "server-control.jsonl",
            "observation": base / "observation.json",
        },
        profile_ids,
        warmup_ids,
    )


def _profile_roots(profile_ids: list[str]) -> dict[str, tuple[str, str]]:
    return {request_id: ("001.json", "root-1") for request_id in profile_ids}


class TestB2AIPerfAdapter(unittest.TestCase):
    def test_zero_warmup_is_explicit_and_still_requires_warmup_phase(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            paths, profile_ids, warmup_ids = _fixture(
                Path(temporary_dir), warmup_ids=[]
            )
            observation = create_observation(
                arm="capture",
                artifact_dir=paths["artifact_dir"],
                client_log_path=paths["client_log"],
                server_log_source=paths["server_log"],
                raw_log_path=paths["raw"],
                business_log_path=paths["business"],
                control_log_path=paths["control"],
                observation_path=paths["observation"],
                expected_profile_ids=profile_ids,
                expected_warmup_ids=warmup_ids,
                profile_root_by_id=_profile_roots(profile_ids),
                selected_sessions=["001.json"],
                root_admissions=[
                    {"source_file": "001.json", "logical_root_id": "root-1"}
                ],
            )

            self.assertEqual(observation["skips"], 0)
            self.assertEqual(observation["phase_drains"]["warmup"]["sent"], 0)
            self.assertEqual(observation["phase_drains"]["warmup"]["completed"], 0)
            self.assertEqual(observation["warmup_request_ids"], [])
            self.assertEqual(
                observation["request_log_partition"]["control_request_finished_count"],
                1,
            )
            self.assertEqual(
                paths["control"].read_text(encoding="utf-8").count("HEALTH_CHECK_"), 1
            )

    def test_skip_field_must_be_present_boolean_false_and_not_none(self) -> None:
        for value in ("missing", None, True):
            with (
                self.subTest(value=value),
                tempfile.TemporaryDirectory() as temporary_dir,
            ):
                paths, profile_ids, warmup_ids = _fixture(Path(temporary_dir))
                export = paths["artifact_dir"] / "profile_export.jsonl"
                rows = [json.loads(line) for line in export.read_text().splitlines()]
                metadata = rows[0]["metadata"]
                if value == "missing":
                    del metadata["context_overflow_skip"]
                else:
                    metadata["context_overflow_skip"] = value
                export.write_text("".join(json.dumps(row) + "\n" for row in rows))

                with self.assertRaisesRegex(
                    AIPerfAdapterError, "skip evidence|overflow skip"
                ):
                    inspect_aiperf_artifacts(
                        artifact_dir=paths["artifact_dir"],
                        client_log_path=paths["client_log"],
                        expected_profile_ids=profile_ids,
                        expected_warmup_ids=warmup_ids,
                        profile_root_by_id=_profile_roots(profile_ids),
                    )

    def test_missing_duplicate_or_extra_request_ids_fail_closed(self) -> None:
        for mutation in ("missing", "duplicate", "extra"):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as temporary_dir,
            ):
                paths, profile_ids, warmup_ids = _fixture(Path(temporary_dir))
                export = paths["artifact_dir"] / "profile_export.jsonl"
                rows = [json.loads(line) for line in export.read_text().splitlines()]
                if mutation == "missing":
                    rows.pop()
                elif mutation == "duplicate":
                    rows.append(dict(rows[0]))
                else:
                    extra = dict(rows[0])
                    extra["metadata"] = dict(
                        extra["metadata"], x_request_id="unexpected"
                    )
                    rows.append(extra)
                export.write_text("".join(json.dumps(row) + "\n" for row in rows))

                with self.assertRaisesRegex(AIPerfAdapterError, "IDs differ|IDs"):
                    inspect_aiperf_artifacts(
                        artifact_dir=paths["artifact_dir"],
                        client_log_path=paths["client_log"],
                        expected_profile_ids=profile_ids,
                        expected_warmup_ids=warmup_ids,
                        profile_root_by_id=_profile_roots(profile_ids),
                    )

    def test_missing_phase_counter_evidence_fails_even_when_metrics_has_null(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            paths, profile_ids, warmup_ids = _fixture(
                Path(temporary_dir), warmup_ids=[]
            )
            log = paths["client_log"]
            text = log.read_text(encoding="utf-8")
            text = "\n".join(
                line
                for line in text.splitlines()
                if "Phase warmup (warmup) sending complete" not in line
            )
            log.write_text(text + "\n", encoding="utf-8")

            with self.assertRaisesRegex(AIPerfAdapterError, "sending and completion"):
                inspect_aiperf_artifacts(
                    artifact_dir=paths["artifact_dir"],
                    client_log_path=log,
                    expected_profile_ids=profile_ids,
                    expected_warmup_ids=warmup_ids,
                    profile_root_by_id=_profile_roots(profile_ids),
                )

    def test_unknown_headerless_finished_event_is_not_discarded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            paths, profile_ids, warmup_ids = _fixture(
                Path(temporary_dir), warmup_ids=[]
            )
            with paths["server_log"].open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {"event": "request.finished", "rid": "other", "headers": {}}
                    )
                    + "\n"
                )

            with self.assertRaisesRegex(AIPerfAdapterError, "no x-request-id"):
                create_observation(
                    arm="capture",
                    artifact_dir=paths["artifact_dir"],
                    client_log_path=paths["client_log"],
                    server_log_source=paths["server_log"],
                    raw_log_path=paths["raw"],
                    business_log_path=paths["business"],
                    control_log_path=paths["control"],
                    observation_path=paths["observation"],
                    expected_profile_ids=profile_ids,
                    expected_warmup_ids=warmup_ids,
                    profile_root_by_id=_profile_roots(profile_ids),
                    selected_sessions=["001.json"],
                    root_admissions=[
                        {"source_file": "001.json", "logical_root_id": "root-1"}
                    ],
                )


if __name__ == "__main__":
    unittest.main()
