from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any


class AIPerfAdapterError(ValueError):
    pass


PHASES = ("warmup", "profiling")
DRAIN_FIELDS = ("sent", "completed", "errors", "skips", "cancellations", "pending")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AIPerfAdapterError(
            f"cannot read AIPerf artifact {path}: {error}"
        ) from error


def _artifact(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _request_id(record: dict[str, Any], label: str) -> str:
    value = record.get("rid")
    if not isinstance(value, str) or not value:
        raise AIPerfAdapterError(f"{label} request.finished has no rid")
    headers = record.get("headers")
    if not isinstance(headers, dict):
        raise AIPerfAdapterError(f"{label} request.finished has no headers")
    normalized = {str(key).lower(): value for key, value in headers.items()}
    request_id = normalized.get("x-request-id")
    occurrence = normalized.get("x-b2-occurrence")
    if not isinstance(request_id, str) or not request_id:
        raise AIPerfAdapterError(f"{label} request.finished has no x-request-id header")
    if not isinstance(occurrence, str) or not occurrence:
        raise AIPerfAdapterError(
            f"{label} request.finished has no x-b2-occurrence header"
        )
    return request_id


def partition_sglang_log(
    source: Path, raw_target: Path, business_target: Path, control_target: Path
) -> dict[str, Any]:
    """Preserve the full server log and split only positively identified health checks."""
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise AIPerfAdapterError(
            f"cannot read configured SGLang request log {source}: {error}"
        ) from error

    business: list[bytes] = []
    controls: list[bytes] = []
    business_ids: list[str] = []
    control_finished = 0
    for line_number, line in enumerate(raw.splitlines(keepends=True), start=1):
        if not line.strip():
            business.append(line)
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise AIPerfAdapterError(
                f"invalid JSON in SGLang log {source}:{line_number}"
            ) from error
        if not isinstance(record, dict):
            raise AIPerfAdapterError(
                f"non-object SGLang event in {source}:{line_number}"
            )
        rid = record.get("rid")
        if isinstance(rid, str) and rid.startswith("HEALTH_CHECK_"):
            controls.append(line)
            control_finished += int(record.get("event") == "request.finished")
            continue
        if record.get("event") == "request.finished":
            business_ids.append(_request_id(record, f"{source}:{line_number}"))
        business.append(line)

    for path, content in (
        (raw_target, raw),
        (business_target, b"".join(business)),
        (control_target, b"".join(controls)),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return {
        "raw": _artifact(raw_target),
        "business": _artifact(business_target),
        "control_sidecar": _artifact(control_target),
        "business_request_finished_ids": business_ids,
        "business_request_finished_count": len(business_ids),
        "control_request_finished_count": control_finished,
    }


def _parse_int_field(text: str, key: str, label: str) -> int:
    match = re.search(rf"\b{re.escape(key)}=(\d+)\b", text)
    if match is None:
        raise AIPerfAdapterError(f"AIPerf client log has no {label} {key} count")
    return int(match.group(1))


def _phase_log_counts(client_log_path: Path) -> dict[str, dict[str, int]]:
    try:
        lines = client_log_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise AIPerfAdapterError(
            f"cannot read AIPerf client log {client_log_path}: {error}"
        ) from error

    sending: dict[str, tuple[int, int]] = {}
    completion: dict[str, tuple[int, int, int]] = {}
    stats: dict[str, dict[str, int]] = {}
    for line in lines:
        send_match = re.search(
            r"Phase .*?\((warmup|profiling)\) sending complete \| sent=(\d+), completed=(\d+), in_flight=(\d+)",
            line,
        )
        if send_match:
            phase, sent, interim_completed, _in_flight = send_match.groups()
            sending[phase] = (int(sent), int(interim_completed))
        complete_match = re.search(
            r"Phase .*?\((warmup|profiling)\) complete \| completed=(\d+), cancelled=(\d+), errors=(\d+)",
            line,
        )
        if complete_match:
            phase, completed, cancelled, errors = complete_match.groups()
            completion[phase] = (int(completed), int(cancelled), int(errors))
        if "PhaseRecordsStats(" in line:
            phase_match = re.search(r"\bphase_kind='(warmup|profiling)'", line)
            if phase_match:
                phase = phase_match.group(1)
                field_values: dict[str, int] = {}
                for field, aliases in {
                    "completed": ("final_requests_completed",),
                    "cancelled": ("final_requests_cancelled",),
                    "errors": ("final_request_errors",),
                }.items():
                    field_values[field] = _parse_int_field(
                        line, aliases[0], f"{phase} PhaseRecordsStats"
                    )
                stats[phase] = field_values

    if set(sending) != set(PHASES) or set(completion) != set(PHASES):
        raise AIPerfAdapterError(
            "AIPerf client log must contain sending and completion records for warmup and profiling"
        )
    if "profiling" not in stats:
        raise AIPerfAdapterError(
            "AIPerf client log has no final profiling PhaseRecordsStats counters"
        )

    result: dict[str, dict[str, int]] = {}
    for phase in PHASES:
        sent, _interim_completed = sending[phase]
        completed, cancelled, errors = completion[phase]
        if phase in stats and stats[phase] != {
            "completed": completed,
            "cancelled": cancelled,
            "errors": errors,
        }:
            raise AIPerfAdapterError(f"AIPerf {phase} final log counters disagree")
        pending = sent - completed - cancelled - errors
        if pending < 0:
            raise AIPerfAdapterError(
                f"AIPerf {phase} final counters exceed sent requests"
            )
        result[phase] = {
            "sent": sent,
            "completed": completed,
            "errors": errors,
            "cancellations": cancelled,
            "pending": pending,
        }
    return result


def _phase_drains(
    artifact_dir: Path, client_log_path: Path
) -> tuple[dict[str, dict[str, int]], dict[str, Path]]:
    manifest_path = artifact_dir / "phase_manifest.json"
    manifest = _read_json(manifest_path)
    phases = manifest.get("phases") if isinstance(manifest, dict) else None
    if not isinstance(phases, list) or not phases:
        raise AIPerfAdapterError("AIPerf phase_manifest.json has no phase entries")
    metrics_paths: dict[str, Path] = {}
    declared_drains: dict[str, dict[str, int]] = {}
    for phase in phases:
        if not isinstance(phase, dict):
            raise AIPerfAdapterError(
                "AIPerf phase manifest contains a non-object phase"
            )
        kind = phase.get("phase_kind")
        if kind not in PHASES or kind in metrics_paths:
            raise AIPerfAdapterError(
                f"AIPerf phase manifest has unsupported or duplicate phase kind: {kind!r}"
            )
        metrics_rel = phase.get("metrics_json")
        if not isinstance(metrics_rel, str) or not metrics_rel:
            raise AIPerfAdapterError(
                f"AIPerf {kind} phase has no metrics_json artifact"
            )
        metrics_path = (artifact_dir / metrics_rel).resolve()
        if (
            not metrics_path.is_relative_to(artifact_dir.resolve())
            or not metrics_path.is_file()
        ):
            raise AIPerfAdapterError(
                f"AIPerf {kind} metrics_json path is invalid: {metrics_rel}"
            )
        metrics_paths[kind] = metrics_path
        drain = phase.get("request_drain")
        if drain is not None:
            if not isinstance(drain, dict):
                raise AIPerfAdapterError(
                    f"AIPerf {kind} request_drain must be an object"
                )
            values = {
                "sent": drain.get("sent"),
                "completed": drain.get("completed"),
                "errors": drain.get("errors"),
                "cancellations": drain.get("cancellations"),
                "pending": drain.get("pending"),
            }
            if any(type(value) is not int or value < 0 for value in values.values()):
                raise AIPerfAdapterError(
                    f"AIPerf {kind} request_drain has incomplete counters"
                )
            declared_drains[kind] = values
    if set(metrics_paths) != set(PHASES):
        raise AIPerfAdapterError(
            "AIPerf phase_manifest.json must include warmup and profiling phases"
        )

    counts = _phase_log_counts(client_log_path)
    for phase, declared in declared_drains.items():
        if declared != counts[phase]:
            raise AIPerfAdapterError(
                f"AIPerf {phase} manifest drain disagrees with client log"
            )
    return counts, metrics_paths


def _exported_request_ids(artifact_dir: Path) -> dict[str, list[str]]:
    value = _read_json(artifact_dir / "outputs.json")
    if not isinstance(value, dict):
        raise AIPerfAdapterError("AIPerf outputs.json must be an object")
    result: dict[str, list[str]] = {"profiling": [], "warmup": []}
    for key, phase in (("data", "profiling"), ("warmup", "warmup")):
        rows = value.get(key)
        if not isinstance(rows, list):
            raise AIPerfAdapterError(f"AIPerf outputs.json has no {key} list")
        for row in rows:
            if not isinstance(row, dict) or row.get("benchmark_phase") != phase:
                raise AIPerfAdapterError(
                    f"AIPerf outputs.json has an invalid {phase} row"
                )
            request_id = row.get("x_request_id")
            if not isinstance(request_id, str) or not request_id:
                raise AIPerfAdapterError(
                    f"AIPerf outputs.json has a {phase} row without x_request_id"
                )
            result[phase].append(request_id)
    return result


def _validate_per_request_export(
    artifact_dir: Path,
    expected_ids: dict[str, list[str]],
    profile_root_by_id: dict[str, tuple[str, str]],
) -> tuple[Path, dict[str, list[str]], list[dict[str, str]]]:
    path = artifact_dir / "profile_export.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise AIPerfAdapterError(
            f"cannot read AIPerf per-request export {path}: {error}"
        ) from error
    found: dict[str, list[str]] = {phase: [] for phase in PHASES}
    profile_timestamps: list[tuple[int, str]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise AIPerfAdapterError(
                f"invalid AIPerf per-request JSON in {path}:{line_number}"
            ) from error
        metadata = row.get("metadata") if isinstance(row, dict) else None
        if not isinstance(metadata, dict):
            raise AIPerfAdapterError(
                f"AIPerf per-request row lacks metadata at {path}:{line_number}"
            )
        phase = metadata.get("benchmark_phase")
        request_id = metadata.get("x_request_id")
        if phase not in PHASES or not isinstance(request_id, str) or not request_id:
            raise AIPerfAdapterError(
                f"AIPerf per-request row lacks phase or x_request_id at {path}:{line_number}"
            )
        if type(metadata.get("context_overflow_skip")) is not bool:
            raise AIPerfAdapterError(
                f"AIPerf per-request row has no boolean skip evidence at {path}:{line_number}"
            )
        if metadata["context_overflow_skip"] is not False:
            raise AIPerfAdapterError(
                f"AIPerf recorded a context-overflow skip for {request_id}"
            )
        if metadata.get("was_cancelled") is not False:
            raise AIPerfAdapterError(
                f"AIPerf per-request row is missing explicit non-cancelled evidence for {request_id}"
            )
        if row.get("error") not in (None, False, ""):
            raise AIPerfAdapterError(
                f"AIPerf per-request row reports an error for {request_id}"
            )
        found[phase].append(request_id)
        if phase == "profiling":
            timestamp = metadata.get("request_start_ns")
            if type(timestamp) is not int or timestamp < 0:
                raise AIPerfAdapterError(
                    f"AIPerf profiling row has no valid request_start_ns for {request_id}"
                )
            profile_timestamps.append((timestamp, request_id))

    for phase, expected in expected_ids.items():
        if found[phase] != expected and (
            len(found[phase]) != len(expected)
            or set(found[phase]) != set(expected)
            or len(set(found[phase])) != len(found[phase])
        ):
            raise AIPerfAdapterError(
                f"AIPerf raw {phase} request IDs differ from the frozen plan"
            )
    ordered_profile_ids = [
        request_id
        for _, request_id in sorted(profile_timestamps, key=lambda item: item[0])
    ]
    if set(ordered_profile_ids) != set(profile_root_by_id):
        raise AIPerfAdapterError(
            "AIPerf profiling root mapping differs from the frozen request IDs"
        )
    admission_order: list[dict[str, str]] = []
    seen_roots: set[tuple[str, str]] = set()
    for request_id in ordered_profile_ids:
        identity = profile_root_by_id[request_id]
        if identity not in seen_roots:
            seen_roots.add(identity)
            admission_order.append(
                {"source_file": identity[0], "logical_root_id": identity[1]}
            )
    return path, found, admission_order


def inspect_aiperf_artifacts(
    *,
    artifact_dir: Path,
    client_log_path: Path,
    expected_profile_ids: list[str],
    expected_warmup_ids: list[str],
    profile_root_by_id: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    if len(expected_profile_ids) != len(set(expected_profile_ids)) or len(
        expected_warmup_ids
    ) != len(set(expected_warmup_ids)):
        raise AIPerfAdapterError("frozen AIPerf request IDs contain duplicates")
    expected = {"profiling": expected_profile_ids, "warmup": expected_warmup_ids}
    drains, metrics_paths = _phase_drains(artifact_dir, client_log_path)
    exported = _exported_request_ids(artifact_dir)
    raw_path, raw_ids, observed_admissions = _validate_per_request_export(
        artifact_dir, expected, profile_root_by_id
    )
    for phase in PHASES:
        for source_name, ids in (
            ("outputs.json", exported[phase]),
            ("profile_export.jsonl", raw_ids[phase]),
        ):
            if (
                len(ids) != len(set(ids))
                or len(ids) != len(expected[phase])
                or set(ids) != set(expected[phase])
            ):
                raise AIPerfAdapterError(
                    f"AIPerf {source_name} {phase} IDs differ from the frozen plan"
                )
        if drains[phase]["sent"] != len(expected[phase]) or drains[phase][
            "completed"
        ] != len(expected[phase]):
            raise AIPerfAdapterError(
                f"AIPerf {phase} final sent/completed counts differ from frozen IDs"
            )
        if any(drains[phase][key] for key in ("errors", "cancellations", "pending")):
            raise AIPerfAdapterError(
                f"AIPerf {phase} final drain contains errors, cancellations, or pending requests"
            )

    artifact_refs = {
        "phase_manifest": _artifact(artifact_dir / "phase_manifest.json"),
        "outputs": _artifact(artifact_dir / "outputs.json"),
        "per_request_export": _artifact(raw_path),
        "client_log": _artifact(client_log_path),
        "metrics": {phase: _artifact(path) for phase, path in metrics_paths.items()},
    }
    return {
        "drains": drains,
        "request_ids": raw_ids,
        "profiling_root_admissions": observed_admissions,
        "artifacts": artifact_refs,
    }


def create_observation(
    *,
    arm: str,
    artifact_dir: Path,
    client_log_path: Path,
    server_log_source: Path,
    raw_log_path: Path,
    business_log_path: Path,
    control_log_path: Path,
    observation_path: Path,
    expected_profile_ids: list[str],
    expected_warmup_ids: list[str],
    profile_root_by_id: dict[str, tuple[str, str]],
    selected_sessions: list[str],
    root_admissions: list[dict[str, str]],
) -> dict[str, Any]:
    aiperf = inspect_aiperf_artifacts(
        artifact_dir=artifact_dir,
        client_log_path=client_log_path,
        expected_profile_ids=expected_profile_ids,
        expected_warmup_ids=expected_warmup_ids,
        profile_root_by_id=profile_root_by_id,
    )
    if aiperf["profiling_root_admissions"] != root_admissions:
        raise AIPerfAdapterError(
            "AIPerf first-seen profiling roots differ from frozen admission order"
        )
    partition = partition_sglang_log(
        server_log_source, raw_log_path, business_log_path, control_log_path
    )
    expected_all = expected_profile_ids + expected_warmup_ids
    actual_ids = partition["business_request_finished_ids"]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_all):
        raise AIPerfAdapterError(
            "SGLang business request IDs differ from the complete frozen phase ledgers"
        )

    totals = {key: 0 for key in DRAIN_FIELDS}
    for values in aiperf["drains"].values():
        for key in ("sent", "completed", "errors", "cancellations", "pending"):
            totals[key] += values[key]
    totals["skips"] = 0
    if totals["sent"] != len(expected_all) or totals["completed"] != len(expected_all):
        raise AIPerfAdapterError(
            "AIPerf sent/completed counters do not cover all frozen request IDs"
        )
    if partition["business_request_finished_count"] != len(expected_all):
        raise AIPerfAdapterError(
            "SGLang business finished-event count differs from frozen request IDs"
        )

    run_hash = hashlib.sha256(
        json.dumps(aiperf["artifacts"], sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    observation = {
        "schema_version": 1,
        "arm": arm,
        "run_id": f"{arm}-{run_hash[:16]}",
        "loaded_sessions": selected_sessions,
        "profiling_root_admissions": root_admissions,
        "request_ids": expected_profile_ids,
        "warmup_request_ids": expected_warmup_ids,
        **totals,
        "phase_drains": aiperf["drains"],
        "skip_evidence": {
            "count": 0,
            "basis": "every frozen ID appears once in outputs.json and profile_export.jsonl with context_overflow_skip=false",
            "per_request_artifact": aiperf["artifacts"]["per_request_export"],
        },
        "aiperf_artifacts": aiperf["artifacts"],
        "request_log_partition": {
            key: partition[key]
            for key in (
                "raw",
                "business",
                "control_sidecar",
                "business_request_finished_count",
                "control_request_finished_count",
            )
        },
    }
    observation_path.parent.mkdir(parents=True, exist_ok=True)
    observation_path.write_text(
        json.dumps(observation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return observation


__all__ = [
    "AIPerfAdapterError",
    "create_observation",
    "inspect_aiperf_artifacts",
    "partition_sglang_log",
]
