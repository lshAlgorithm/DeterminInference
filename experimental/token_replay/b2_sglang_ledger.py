from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from b2_aiperf_adapter import (
    AIPerfAdapterError,
    inspect_aiperf_artifacts,
    partition_sglang_log,
)


ARMS = ("capture", "generated", "replay")
PROFILE_ROOT_ADMISSION_COUNT = 4
OCCURRENCE_FIELDS = (
    "source_file",
    "source_row",
    "phase",
    "logical_root_id",
    "tree_occurrence_id",
    "turn",
    "reset_ordinal",
    "cache_salt_class",
)
DRAIN_FIELDS = (
    "sent",
    "completed",
    "errors",
    "skips",
    "cancellations",
    "pending",
)


class RuntimeLedgerError(ValueError):
    pass


def _nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _validate_server_config(config: Any) -> None:
    if not isinstance(config, dict):
        raise RuntimeLedgerError("server_logging must be an object")
    expected = {
        "enabled": True,
        "format": "json",
        "level": 3,
        "incremental_streaming_output": False,
        "request_id_header_logged": True,
        "dedicated_target": True,
    }
    for key, value in expected.items():
        if config.get(key) != value or type(config.get(key)) is not type(value):
            raise RuntimeLedgerError(
                f"server_logging.{key} must be {value!r} for complete request.finished evidence"
            )
    log_exceeded_ms = config.get("log_exceeded_ms")
    if type(log_exceeded_ms) is not int or log_exceeded_ms > 0:
        raise RuntimeLedgerError("server_logging.log_exceeded_ms must be <= 0")


def _resolve_log_files(values: Any, base_dir: Path, arm: str) -> list[tuple[Path, str]]:
    if not isinstance(values, list) or not values:
        raise RuntimeLedgerError(
            f"request_logs.{arm} must list at least one hashed file"
        )
    files: dict[Path, str] = {}
    for entry in values:
        if not isinstance(entry, dict):
            raise RuntimeLedgerError(
                f"request_logs.{arm} entries must contain path and sha256"
            )
        raw_path = entry.get("path")
        expected_hash = entry.get("sha256")
        if not isinstance(raw_path, str) or not raw_path:
            raise RuntimeLedgerError(f"request_logs.{arm} contains an invalid path")
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(char not in "0123456789abcdef" for char in expected_hash)
        ):
            raise RuntimeLedgerError(f"request_logs.{arm} contains an invalid SHA-256")
        path = Path(raw_path)
        if not path.is_absolute():
            path = base_dir / path
        if path.is_file():
            resolved = path.resolve()
            if resolved in files:
                raise RuntimeLedgerError(
                    f"request_logs.{arm} repeats a log path: {resolved}"
                )
            files[resolved] = expected_hash
        elif path.is_dir():
            raise RuntimeLedgerError(
                f"request_logs.{arm} entries must identify hashed files, not directories: {path}"
            )
        else:
            raise RuntimeLedgerError(f"request log path does not exist: {path}")
    if not files:
        raise RuntimeLedgerError(f"request_logs.{arm} resolved to no log files")
    for path, expected_hash in files.items():
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeLedgerError(f"request log SHA-256 mismatch: {path}")
    return sorted(files.items())


def _read_hashed_json(
    entry: Any, base_dir: Path, label: str
) -> tuple[dict[str, Any], dict[str, str]]:
    if not isinstance(entry, dict):
        raise RuntimeLedgerError(f"{label} must contain path and sha256")
    raw_path = entry.get("path")
    expected_hash = entry.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeLedgerError(f"{label} has an invalid path")
    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or any(char not in "0123456789abcdef" for char in expected_hash)
    ):
        raise RuntimeLedgerError(f"{label} has an invalid SHA-256")
    path = Path(raw_path)
    if not path.is_absolute():
        path = base_dir / path
    if not path.is_file():
        raise RuntimeLedgerError(f"{label} file does not exist: {path}")
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeLedgerError(f"{label} SHA-256 mismatch: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeLedgerError(f"cannot read {label} JSON: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeLedgerError(f"{label} JSON must be an object")
    return value, {"path": str(path.resolve()), "sha256": actual_hash}


def _resolve_hashed_artifact(
    entry: Any, base_dir: Path, label: str
) -> tuple[Path, str]:
    if not isinstance(entry, dict):
        raise RuntimeLedgerError(f"{label} must contain path and sha256")
    raw_path, expected_hash = entry.get("path"), entry.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeLedgerError(f"{label} has an invalid path")
    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or any(char not in "0123456789abcdef" for char in expected_hash)
    ):
        raise RuntimeLedgerError(f"{label} has an invalid SHA-256")
    path = Path(raw_path)
    if not path.is_absolute():
        path = base_dir / path
    if not path.is_file():
        raise RuntimeLedgerError(f"{label} file does not exist: {path}")
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeLedgerError(f"{label} SHA-256 mismatch: {path}")
    return path.resolve(), actual_hash


def _canonical_json_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _decode_occurrence_header(value: Any) -> dict[str, Any]:
    """Decode the canonical URL-safe base64 occurrence header emitted by AIPerf."""
    if (
        not isinstance(value, str)
        or not value
        or any(
            char
            not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
            for char in value
        )
    ):
        raise RuntimeLedgerError("invalid x-b2-occurrence header encoding")
    padded = value + "=" * (-len(value) % 4)
    try:
        payload = base64.urlsafe_b64decode(padded)
        occurrence = json.loads(payload)
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeLedgerError("invalid x-b2-occurrence header payload") from error
    if not isinstance(occurrence, dict):
        raise RuntimeLedgerError("x-b2-occurrence header payload must be an object")
    canonical = (
        base64.urlsafe_b64encode(
            json.dumps(occurrence, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        .decode("ascii")
        .rstrip("=")
    )
    if canonical != value:
        raise RuntimeLedgerError("x-b2-occurrence header is not canonical")
    return occurrence


def _read_finished_events(
    paths: list[tuple[Path, str]], arm: str
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path, _expected_hash in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise RuntimeLedgerError(
                f"cannot read request log {path}: {error}"
            ) from error
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeLedgerError(
                    f"invalid JSON in {path}:{line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise RuntimeLedgerError(
                    f"non-object request log entry in {path}:{line_number}"
                )
            if value.get("event") == "request.finished":
                events.append(value)
    if not events:
        raise RuntimeLedgerError(f"no request.finished events found for {arm}")
    return events


def _request_specs(
    value: Any, arm: str, expected_phase: str, allow_empty: bool = False
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise RuntimeLedgerError(
            f"{expected_phase}_requests_by_arm.{arm} must be a non-empty frozen request list"
        )
    specs: dict[str, dict[str, Any]] = {}
    identities: set[tuple[Any, ...]] = set()
    for item in value:
        if not isinstance(item, dict):
            raise RuntimeLedgerError(
                f"{expected_phase}_requests_by_arm.{arm} contains a non-object entry"
            )
        request_id = item.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise RuntimeLedgerError(
                f"{expected_phase}_requests_by_arm.{arm} has an invalid request_id"
            )
        if not request_id.endswith(f"-{expected_phase}"):
            raise RuntimeLedgerError(
                f"{expected_phase}_requests_by_arm.{arm} request_id is not phase-bound"
            )
        if request_id in specs:
            raise RuntimeLedgerError(
                f"{expected_phase}_requests_by_arm.{arm} has duplicate request_id {request_id}"
            )
        if not set(OCCURRENCE_FIELDS).issubset(item):
            raise RuntimeLedgerError(
                f"{expected_phase}_requests_by_arm.{arm} is missing occurrence identity fields"
            )
        for key in (
            "source_file",
            "phase",
            "logical_root_id",
            "tree_occurrence_id",
            "cache_salt_class",
        ):
            if not isinstance(item[key], str) or not item[key]:
                raise RuntimeLedgerError(
                    f"{expected_phase}_requests_by_arm.{arm} has invalid {key}"
                )
        for key in ("source_row", "turn", "reset_ordinal"):
            if not _nonnegative_int(item[key]):
                raise RuntimeLedgerError(
                    f"{expected_phase}_requests_by_arm.{arm} has invalid {key}"
                )
        if item["phase"] != expected_phase:
            raise RuntimeLedgerError(
                f"{expected_phase}_requests_by_arm.{arm} phase must be {expected_phase!r}"
            )
        occurrence = {key: item[key] for key in OCCURRENCE_FIELDS}
        identity = tuple(occurrence[key] for key in OCCURRENCE_FIELDS)
        if identity in identities:
            raise RuntimeLedgerError(
                f"{expected_phase}_requests_by_arm.{arm} contains a duplicate occurrence"
            )
        identities.add(identity)
        specs[request_id] = occurrence
    return specs


def _event_from_log(
    record: dict[str, Any], arm: str, specs: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    headers = record.get("headers")
    if not isinstance(headers, dict):
        raise RuntimeLedgerError(
            f"{arm} request.finished event has no logged request headers"
        )
    normalized_headers = {str(key).lower(): value for key, value in headers.items()}
    request_id = normalized_headers.get("x-request-id")
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeLedgerError(
            f"{arm} request.finished event has no x-request-id header"
        )
    if request_id not in specs:
        raise RuntimeLedgerError(
            f"{arm} request.finished contains an unexpected x-request-id: {request_id}"
        )
    raw_occurrence = normalized_headers.get("x-b2-occurrence")
    if raw_occurrence is None:
        raise RuntimeLedgerError(
            f"{arm} request.finished event has no x-b2-occurrence header"
        )
    occurrence = _decode_occurrence_header(raw_occurrence)
    expected_occurrence = specs[request_id]
    if occurrence != expected_occurrence:
        raise RuntimeLedgerError(
            f"{arm} request.finished x-b2-occurrence differs from the frozen mapping"
        )

    out = record.get("out")
    meta_info = out.get("meta_info") if isinstance(out, dict) else None
    if not isinstance(out, dict) or not isinstance(meta_info, dict):
        raise RuntimeLedgerError(f"{arm} request.finished event has no out.meta_info")
    server_request_id = record.get("rid")
    if (
        not isinstance(server_request_id, str)
        or meta_info.get("id") != server_request_id
    ):
        raise RuntimeLedgerError(
            f"{arm} request.finished rid differs from out.meta_info.id"
        )

    output_ids = out.get("output_ids")
    if (
        not isinstance(output_ids, list)
        or not output_ids
        or any(type(token) is not int for token in output_ids)
        or "..." in output_ids
    ):
        raise RuntimeLedgerError(
            f"{arm} request.finished has missing, truncated, or invalid output_ids"
        )
    prompt_tokens = meta_info.get("prompt_tokens")
    cache_hit_tokens = meta_info.get("cached_tokens")
    total_retractions = meta_info.get("total_retractions")
    if not _nonnegative_int(prompt_tokens) or not _nonnegative_int(cache_hit_tokens):
        raise RuntimeLedgerError(
            f"{arm} request.finished has invalid prompt_tokens or cached_tokens"
        )
    if not _nonnegative_int(total_retractions):
        raise RuntimeLedgerError(
            f"{arm} request.finished has invalid total_retractions"
        )
    if cache_hit_tokens > prompt_tokens:
        raise RuntimeLedgerError(f"{arm} cached_tokens exceeds prompt_tokens")
    if len(output_ids) != meta_info.get("completion_tokens"):
        raise RuntimeLedgerError(
            f"{arm} output_ids length differs from completion_tokens"
        )

    event = {
        "event": "request.finished",
        "arm": arm,
        "request_id": request_id,
        **occurrence,
        "output_ids": output_ids,
        "output_length": len(output_ids),
        "cache_hit_tokens": cache_hit_tokens,
        "new_prefill_tokens": prompt_tokens - cache_hit_tokens,
        "total_retractions": total_retractions,
        "server_request_id": server_request_id,
    }
    return event


def materialize_capture_reference(
    reference: dict[str, Any], frozen_plan: dict[str, Any], log_path: Path
) -> dict[str, Any]:
    """Build a Qwen reference from capture request.finished output vectors."""
    if not log_path.is_file():
        raise RuntimeLedgerError(f"capture request log does not exist: {log_path}")
    profile_specs = _request_specs(
        frozen_plan.get("requests_by_arm", {}).get("capture"), "capture", "profiling"
    )
    warmup_specs = _request_specs(
        frozen_plan.get("warmup_requests_by_arm", {}).get("capture"),
        "capture",
        "warmup",
        allow_empty=True,
    )
    specs = {**profile_specs, **warmup_specs}
    events = _read_finished_events([(log_path, "")], "capture")
    captured: dict[tuple[str, int], list[int]] = {}
    for record in events:
        event = _event_from_log(record, "capture", specs)
        key = (event["source_file"], event["source_row"])
        if key in captured:
            raise RuntimeLedgerError(
                f"capture reference contains duplicate source row {key[0]} row {key[1]}"
            )
        captured[key] = list(event["output_ids"])

    rows = reference.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeLedgerError("capture-derived reference source has no rows")
    materialized_rows: list[dict[str, Any]] = []
    expected_keys: set[tuple[str, int]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeLedgerError(
                "capture-derived reference source has a non-object row"
            )
        session, source_row = row.get("session"), row.get("request")
        if not isinstance(session, str) or type(source_row) is not int:
            raise RuntimeLedgerError(
                "capture-derived reference row has an invalid locator"
            )
        key = (session, source_row)
        expected_keys.add(key)
        output_ids = captured.get(key)
        if output_ids is None:
            raise RuntimeLedgerError(
                f"capture reference is missing {session} row {source_row}"
            )
        source_output_ids = row.get("output_ids")
        if not isinstance(source_output_ids, list) or len(source_output_ids) != len(
            output_ids
        ):
            raise RuntimeLedgerError(
                f"capture output length differs from source for {session} row {source_row}"
            )
        materialized_rows.append({**row, "output_ids": output_ids})
    if set(captured) != expected_keys:
        extra = sorted(set(captured) - expected_keys)
        raise RuntimeLedgerError(
            f"capture log contains unmapped source rows: {extra[:3]}"
        )
    return {
        **reference,
        "reference_mode": "capture_derived",
        "source_reference_sha256": _canonical_json_sha256(reference),
        "rows": materialized_rows,
    }


def build_event_bundle(
    manifest: dict[str, Any], base_dir: Path, reference: dict[str, Any]
) -> dict[str, Any]:
    if manifest.get("schema_version") != 1:
        raise RuntimeLedgerError("runtime manifest schema_version must be 1")
    _validate_server_config(manifest.get("server_logging"))

    plan, plan_provenance = _read_hashed_json(
        manifest.get("frozen_plan"), base_dir, "frozen_plan"
    )
    if plan.get("schema_version") != 1:
        raise RuntimeLedgerError("frozen_plan schema_version must be 1")
    expected_reference_hash = _canonical_json_sha256(reference)
    if plan.get("reference_sha256") != expected_reference_hash:
        raise RuntimeLedgerError(
            "frozen_plan reference SHA-256 does not match the gate reference"
        )
    execution_config_path = None
    execution_config_hash = None
    if manifest.get("execution_config") is not None:
        execution_config, execution_config_provenance = _read_hashed_json(
            manifest["execution_config"], base_dir, "execution_config"
        )
        execution_config_path = Path(execution_config_provenance["path"])
        execution_config_hash = execution_config_provenance["sha256"]
        config_arms = execution_config.get("arms")
        if not isinstance(config_arms, dict) or set(config_arms) != set(ARMS):
            raise RuntimeLedgerError(
                "execution_config must freeze all three arm commands"
            )
        for arm in ARMS:
            spec = config_arms[arm]
            if (
                not isinstance(spec, dict)
                or not isinstance(spec.get("argv"), list)
                or not spec["argv"]
                or not all(isinstance(arg, str) and arg for arg in spec["argv"])
                or not isinstance(spec.get("cwd"), str)
                or not spec["cwd"]
                or not isinstance(spec.get("aiperf_artifact_dir"), str)
                or not spec["aiperf_artifact_dir"]
                or spec.get("gpu_lifecycle_included") is not True
            ):
                raise RuntimeLedgerError(
                    f"execution_config.{arm} does not freeze command, output, and lifecycle timing"
                )
    else:
        raise RuntimeLedgerError("runtime manifest has no hashed execution_config")
    loaded_sessions = plan.get("selected_sessions")
    if (
        not isinstance(loaded_sessions, list)
        or not loaded_sessions
        or not all(isinstance(x, str) and x for x in loaded_sessions)
        or len(set(loaded_sessions)) != len(loaded_sessions)
    ):
        raise RuntimeLedgerError(
            "frozen_plan selected_sessions must be unique session names"
        )

    raw_specs = plan.get("requests_by_arm")
    raw_warmup_specs = plan.get("warmup_requests_by_arm")
    request_logs = manifest.get("request_logs")
    raw_receipts = manifest.get("runner_receipts")
    if not isinstance(raw_specs, dict) or set(raw_specs) != set(ARMS):
        raise RuntimeLedgerError(
            "frozen_plan.requests_by_arm must contain exactly capture, generated, and replay"
        )
    if not isinstance(raw_warmup_specs, dict) or set(raw_warmup_specs) != set(ARMS):
        raise RuntimeLedgerError(
            "frozen_plan.warmup_requests_by_arm must contain exactly capture, generated, and replay"
        )
    if not isinstance(request_logs, dict) or set(request_logs) != set(ARMS):
        raise RuntimeLedgerError(
            "request_logs must contain exactly capture, generated, and replay"
        )
    if not isinstance(raw_receipts, dict) or set(raw_receipts) != set(ARMS):
        raise RuntimeLedgerError(
            "runner_receipts must contain exactly capture, generated, and replay"
        )

    specs_by_arm = {
        arm: _request_specs(raw_specs[arm], arm, "profiling") for arm in ARMS
    }
    warmup_specs_by_arm = {
        arm: _request_specs(raw_warmup_specs[arm], arm, "warmup", allow_empty=True)
        for arm in ARMS
    }
    selected_session_set = set(loaded_sessions)
    for arm, specs in specs_by_arm.items():
        if any(
            spec["source_file"] not in selected_session_set for spec in specs.values()
        ):
            raise RuntimeLedgerError(
                f"frozen_plan.requests_by_arm.{arm} references a session outside selected_sessions"
            )
    for arm, specs in warmup_specs_by_arm.items():
        if any(
            spec["source_file"] not in selected_session_set for spec in specs.values()
        ):
            raise RuntimeLedgerError(
                f"frozen_plan.warmup_requests_by_arm.{arm} references a session outside selected_sessions"
            )
    expected_by_arm: dict[str, Counter[tuple[Any, ...]]] = {}
    for arm, specs in specs_by_arm.items():
        expected_by_arm[arm] = Counter(
            tuple(item[key] for key in OCCURRENCE_FIELDS) for item in specs.values()
        )
    if not (
        expected_by_arm["capture"]
        == expected_by_arm["generated"]
        == expected_by_arm["replay"]
    ):
        raise RuntimeLedgerError(
            "frozen request occurrence mappings are asymmetric across arms"
        )
    expected_warmup_by_arm = {
        arm: Counter(
            tuple(item[key] for key in OCCURRENCE_FIELDS)
            for item in warmup_specs_by_arm[arm].values()
        )
        for arm in ARMS
    }
    if not (
        expected_warmup_by_arm["capture"]
        == expected_warmup_by_arm["generated"]
        == expected_warmup_by_arm["replay"]
    ):
        raise RuntimeLedgerError(
            "frozen warmup occurrence mappings are asymmetric across arms"
        )
    frozen_admissions = plan.get("profiling_root_admissions")
    if (
        not isinstance(frozen_admissions, list)
        or len(frozen_admissions) != PROFILE_ROOT_ADMISSION_COUNT
        or any(
            not isinstance(item, dict)
            or not isinstance(item.get("source_file"), str)
            or not isinstance(item.get("logical_root_id"), str)
            for item in frozen_admissions
        )
    ):
        raise RuntimeLedgerError(
            "frozen_plan must contain exactly four ordered profiling root admissions"
        )
    profile_roots = {
        (item["source_file"], item["logical_root_id"])
        for item in specs_by_arm["capture"].values()
    }
    if len(profile_roots) != PROFILE_ROOT_ADMISSION_COUNT:
        raise RuntimeLedgerError(
            "frozen profiling requests must contain exactly four roots"
        )

    receipts: dict[str, dict[str, Any]] = {}
    receipt_provenance: dict[str, dict[str, str]] = {}
    aiperf_evidence: dict[str, dict[str, Any]] = {}
    partition_evidence: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        receipt, receipt_provenance[arm] = _read_hashed_json(
            raw_receipts[arm], base_dir, f"runner_receipts.{arm}"
        )
        if receipt.get("schema_version") != 1 or receipt.get("arm") != arm:
            raise RuntimeLedgerError(
                f"runner_receipts.{arm} has an invalid schema or arm"
            )
        if receipt.get("frozen_plan_sha256") != plan_provenance["sha256"]:
            raise RuntimeLedgerError(
                f"runner_receipts.{arm} does not reference the frozen plan hash"
            )
        if (
            execution_config_hash is None
            or receipt.get("execution_config_sha256") != execution_config_hash
        ):
            raise RuntimeLedgerError(
                f"runner_receipts.{arm} does not bind the frozen execution config"
            )
        run_id = receipt.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise RuntimeLedgerError(f"runner_receipts.{arm} has no run_id")
        if receipt.get("loaded_sessions") != loaded_sessions:
            raise RuntimeLedgerError(
                f"runner_receipts.{arm}.loaded_sessions differs from the frozen plan"
            )
        request_ids = receipt.get("request_ids")
        if (
            not isinstance(request_ids, list)
            or not all(isinstance(value, str) and value for value in request_ids)
            or len(set(request_ids)) != len(request_ids)
            or set(request_ids) != set(specs_by_arm[arm])
        ):
            raise RuntimeLedgerError(
                f"runner_receipts.{arm}.request_ids differ from the frozen plan"
            )
        warmup_request_ids = receipt.get("warmup_request_ids")
        if (
            not isinstance(warmup_request_ids, list)
            or not all(isinstance(value, str) and value for value in warmup_request_ids)
            or len(set(warmup_request_ids)) != len(warmup_request_ids)
            or set(warmup_request_ids) != set(warmup_specs_by_arm[arm])
        ):
            raise RuntimeLedgerError(
                f"runner_receipts.{arm}.warmup_request_ids differ from the frozen warmup plan"
            )
        admissions = receipt.get("profiling_root_admissions")
        if (
            not isinstance(admissions, list)
            or len(admissions) != PROFILE_ROOT_ADMISSION_COUNT
            or any(
                not isinstance(item, dict)
                or not isinstance(item.get("source_file"), str)
                or item["source_file"] not in selected_session_set
                or not isinstance(item.get("logical_root_id"), str)
                or not item["logical_root_id"]
                for item in admissions
            )
        ):
            raise RuntimeLedgerError(
                f"runner_receipts.{arm}.profiling_root_admissions must contain exactly four roots"
            )
        admission_keys = [
            (item["source_file"], item["logical_root_id"]) for item in admissions
        ]
        if len(set(admission_keys)) != PROFILE_ROOT_ADMISSION_COUNT:
            raise RuntimeLedgerError(
                f"runner_receipts.{arm}.profiling_root_admissions must contain four distinct roots"
            )
        if set(admission_keys) != profile_roots:
            raise RuntimeLedgerError(
                f"runner_receipts.{arm}.profiling_root_admissions differ from profiling request roots"
            )
        if admissions != frozen_admissions:
            raise RuntimeLedgerError(
                f"runner_receipts.{arm} admission order differs from frozen plan"
            )
        drain_receipt = {key: receipt.get(key) for key in DRAIN_FIELDS}
        if any(not _nonnegative_int(value) for value in drain_receipt.values()):
            raise RuntimeLedgerError(f"runner_receipts.{arm} has invalid drain counts")
        expected_sent = len(request_ids) + len(warmup_request_ids)
        if drain_receipt["sent"] != expected_sent:
            raise RuntimeLedgerError(
                f"runner_receipts.{arm}.sent differs from profiling plus warmup request IDs"
            )
        if drain_receipt["completed"] != drain_receipt["sent"] or any(
            drain_receipt[key] != 0
            for key in ("errors", "skips", "cancellations", "pending")
        ):
            raise RuntimeLedgerError(f"runner_receipts.{arm} did not drain cleanly")

        phase_drains = receipt.get("phase_drains")
        if not isinstance(phase_drains, dict) or set(phase_drains) != {
            "warmup",
            "profiling",
        }:
            raise RuntimeLedgerError(
                f"runner_receipts.{arm} must contain warmup and profiling drain counters"
            )
        for phase, expected_count in (
            ("warmup", len(warmup_request_ids)),
            ("profiling", len(request_ids)),
        ):
            values = phase_drains[phase]
            fields = ("sent", "completed", "errors", "cancellations", "pending")
            if not isinstance(values, dict) or any(
                not _nonnegative_int(values.get(key)) for key in fields
            ):
                raise RuntimeLedgerError(
                    f"runner_receipts.{arm}.{phase} has incomplete drain counters"
                )
            if (
                values["sent"] != expected_count
                or values["completed"] != expected_count
                or any(values[key] for key in ("errors", "cancellations", "pending"))
            ):
                raise RuntimeLedgerError(
                    f"runner_receipts.{arm}.{phase} does not match frozen request IDs"
                )
        if (
            sum(phase_drains[phase]["sent"] for phase in phase_drains)
            != drain_receipt["sent"]
        ):
            raise RuntimeLedgerError(
                f"runner_receipts.{arm} per-phase sent counts differ from total drain"
            )

        skip_evidence = receipt.get("skip_evidence")
        if not isinstance(skip_evidence, dict) or skip_evidence.get("count") != 0:
            raise RuntimeLedgerError(
                f"runner_receipts.{arm} has no explicit zero-skip evidence"
            )
        timing = receipt.get("arm_timing")
        gpu_active_seconds = receipt.get("gpu_active_seconds")
        if (
            not isinstance(timing, dict)
            or isinstance(gpu_active_seconds, bool)
            or not isinstance(gpu_active_seconds, (int, float))
            or not math.isfinite(gpu_active_seconds)
            or gpu_active_seconds <= 0
            or timing.get("gpu_active_seconds") != gpu_active_seconds
            or timing.get("elapsed_seconds") != gpu_active_seconds
            or not isinstance(timing.get("clock_semantics"), str)
            or not timing.get("clock_semantics")
            or not isinstance(timing.get("wall_start_utc"), str)
            or not isinstance(timing.get("wall_end_utc"), str)
        ):
            raise RuntimeLedgerError(
                f"runner_receipts.{arm} has no valid monotonic GPU timing receipt"
            )
        mono_start = timing.get("monotonic_start_seconds")
        mono_end = timing.get("monotonic_end_seconds")
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in (mono_start, mono_end)
            )
            or mono_end < mono_start
            or not math.isclose(
                mono_end - mono_start, gpu_active_seconds, rel_tol=1e-9, abs_tol=1e-6
            )
        ):
            raise RuntimeLedgerError(
                f"runner_receipts.{arm} monotonic timestamps disagree with GPU duration"
            )
        raw_aiperf = receipt.get("aiperf_artifacts")
        if not isinstance(raw_aiperf, dict) or not isinstance(
            raw_aiperf.get("metrics"), dict
        ):
            raise RuntimeLedgerError(
                f"runner_receipts.{arm} has no AIPerf artifact references"
            )
        artifact_paths: dict[str, Path] = {}
        for key in ("phase_manifest", "outputs", "per_request_export", "client_log"):
            artifact_paths[key], _ = _resolve_hashed_artifact(
                raw_aiperf.get(key),
                base_dir,
                f"runner_receipts.{arm}.aiperf_artifacts.{key}",
            )
        metrics_refs = raw_aiperf["metrics"]
        if set(metrics_refs) != {"warmup", "profiling"}:
            raise RuntimeLedgerError(
                f"runner_receipts.{arm} AIPerf metrics do not cover both phases"
            )
        for phase in ("warmup", "profiling"):
            artifact_paths[f"metrics.{phase}"], _ = _resolve_hashed_artifact(
                metrics_refs[phase],
                base_dir,
                f"runner_receipts.{arm}.aiperf_artifacts.metrics.{phase}",
            )
        artifact_dir = artifact_paths["phase_manifest"].parent
        if any(
            artifact_paths[key].parent != artifact_dir
            for key in ("outputs", "per_request_export")
        ):
            raise RuntimeLedgerError(
                f"runner_receipts.{arm} AIPerf artifacts are not in one output directory"
            )
        profile_root_by_id = {
            request_id: (
                specs_by_arm[arm][request_id]["source_file"],
                specs_by_arm[arm][request_id]["logical_root_id"],
            )
            for request_id in specs_by_arm[arm]
        }
        try:
            aiperf_evidence[arm] = inspect_aiperf_artifacts(
                artifact_dir=artifact_dir,
                client_log_path=artifact_paths["client_log"],
                expected_profile_ids=list(specs_by_arm[arm]),
                expected_warmup_ids=list(warmup_specs_by_arm[arm]),
                profile_root_by_id=profile_root_by_id,
            )
        except AIPerfAdapterError as error:
            raise RuntimeLedgerError(
                f"{arm} AIPerf artifacts failed independent validation: {error}"
            ) from error
        if aiperf_evidence[arm]["profiling_root_admissions"] != admissions:
            raise RuntimeLedgerError(
                f"runner_receipts.{arm} admissions differ from AIPerf first-seen roots"
            )
        if aiperf_evidence[arm]["drains"] != phase_drains:
            raise RuntimeLedgerError(
                f"runner_receipts.{arm} phase drains differ from AIPerf raw evidence"
            )
        if skip_evidence.get("per_request_artifact") != raw_aiperf.get(
            "per_request_export"
        ):
            raise RuntimeLedgerError(
                f"runner_receipts.{arm} skip evidence does not bind AIPerf raw export"
            )

        raw_partition = receipt.get("request_log_partition")
        if not isinstance(raw_partition, dict) or not {
            "raw",
            "business",
            "control_sidecar",
        }.issubset(raw_partition):
            raise RuntimeLedgerError(
                f"runner_receipts.{arm} has incomplete SGLang log partition references"
            )
        partition_paths = {
            key: _resolve_hashed_artifact(
                raw_partition[key],
                base_dir,
                f"runner_receipts.{arm}.request_log_partition.{key}",
            )[0]
            for key in ("raw", "business", "control_sidecar")
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_base = Path(temp_dir)
            try:
                rebuilt = partition_sglang_log(
                    partition_paths["raw"],
                    temp_base / "raw.jsonl",
                    temp_base / "business.jsonl",
                    temp_base / "control.jsonl",
                )
            except AIPerfAdapterError as error:
                raise RuntimeLedgerError(
                    f"{arm} raw SGLang request log is invalid: {error}"
                ) from error
            for key, rebuilt_path in (
                ("raw", temp_base / "raw.jsonl"),
                ("business", temp_base / "business.jsonl"),
                ("control_sidecar", temp_base / "control.jsonl"),
            ):
                if partition_paths[key].read_bytes() != rebuilt_path.read_bytes():
                    raise RuntimeLedgerError(
                        f"{arm} {key} SGLang log is not a faithful raw-log partition"
                    )
        partition_evidence[arm] = {
            **rebuilt,
            "paths": {key: str(path) for key, path in partition_paths.items()},
        }
        receipts[arm] = {
            "run_id": run_id,
            "loaded_sessions": receipt["loaded_sessions"],
            "profiling_root_admissions": admissions,
            "request_ids": request_ids,
            "warmup_request_ids": warmup_request_ids,
            "drain": drain_receipt,
            "phase_drains": phase_drains,
            "arm_timing": timing,
            "gpu_active_seconds": gpu_active_seconds,
        }

    admission_keys = {
        arm: [
            (item.get("source_file"), item.get("logical_root_id"))
            for item in receipts[arm]["profiling_root_admissions"]
            if isinstance(item, dict)
        ]
        for arm in ARMS
    }
    if not (
        admission_keys["capture"]
        == admission_keys["generated"]
        == admission_keys["replay"]
    ):
        raise RuntimeLedgerError(
            "runner root-admission receipts are asymmetric across arms"
        )
    run_ids = [receipts[arm]["run_id"] for arm in ARMS]
    if len(set(run_ids)) != len(run_ids):
        raise RuntimeLedgerError(
            "runner_receipts must have distinct run_id values per arm"
        )

    resolved_logs = {
        arm: _resolve_log_files(request_logs[arm], base_dir, arm) for arm in ARMS
    }
    for arm in ARMS:
        business_path = partition_evidence[arm]["paths"]["business"]
        if len(resolved_logs[arm]) != 1 or resolved_logs[arm][0][0] != Path(
            business_path
        ):
            raise RuntimeLedgerError(
                f"{arm} hashed request log is not the verified business sidecar"
            )
    all_paths = [path for paths in resolved_logs.values() for path, _ in paths]
    if len(set(all_paths)) != len(all_paths):
        raise RuntimeLedgerError("a request log file cannot be reused across arms")

    events: list[dict[str, Any]] = []
    warmup_events: list[dict[str, Any]] = []
    for arm in ARMS:
        seen_profile: Counter[str] = Counter()
        seen_warmup: Counter[str] = Counter()
        sglang_root_set: set[tuple[str, str]] = set()
        all_specs = {**specs_by_arm[arm], **warmup_specs_by_arm[arm]}
        for record in _read_finished_events(resolved_logs[arm], arm):
            event = _event_from_log(record, arm, all_specs)
            if event["phase"] == "profiling":
                seen_profile[event["request_id"]] += 1
                sglang_root_set.add((event["source_file"], event["logical_root_id"]))
                events.append(event)
            elif event["phase"] == "warmup":
                seen_warmup[event["request_id"]] += 1
                warmup_events.append(event)
            else:
                raise RuntimeLedgerError(
                    f"{arm} request.finished contains an unsupported phase: {event['phase']!r}"
                )
        missing_profile = sorted(set(specs_by_arm[arm]) - set(seen_profile))
        duplicate_profile = sorted(
            request_id for request_id, count in seen_profile.items() if count != 1
        )
        missing_warmup = sorted(set(warmup_specs_by_arm[arm]) - set(seen_warmup))
        duplicate_warmup = sorted(
            request_id for request_id, count in seen_warmup.items() if count != 1
        )
        if missing_profile:
            raise RuntimeLedgerError(
                f"{arm} profiling request.finished ledger is missing request IDs: {missing_profile}"
            )
        if duplicate_profile:
            raise RuntimeLedgerError(
                f"{arm} profiling request.finished ledger duplicates request IDs: {duplicate_profile}"
            )
        if missing_warmup:
            raise RuntimeLedgerError(
                f"{arm} warmup request.finished ledger is missing request IDs: {missing_warmup}"
            )
        if duplicate_warmup:
            raise RuntimeLedgerError(
                f"{arm} warmup request.finished ledger duplicates request IDs: {duplicate_warmup}"
            )

        if seen_profile != Counter(
            {request_id: 1 for request_id in receipts[arm]["request_ids"]}
        ):
            raise RuntimeLedgerError(
                f"{arm} profiling request.finished IDs differ from the independently recorded runner receipt"
            )
        if seen_warmup != Counter(
            {request_id: 1 for request_id in receipts[arm]["warmup_request_ids"]}
        ):
            raise RuntimeLedgerError(
                f"{arm} warmup request.finished IDs differ from the independently recorded runner receipt"
            )
        expected_root_set = {
            (item["source_file"], item["logical_root_id"])
            for item in specs_by_arm[arm].values()
        }
        if sglang_root_set != expected_root_set:
            raise RuntimeLedgerError(
                f"{arm} SGLang profile roots differ from frozen AIPerf request mapping"
            )

    expected_occurrences = [
        {
            **dict(zip(OCCURRENCE_FIELDS, occurrence, strict=True)),
            "count": count,
        }
        for occurrence, count in sorted(expected_by_arm["capture"].items())
    ]
    expected_warmup_occurrences = [
        {
            **dict(zip(OCCURRENCE_FIELDS, occurrence, strict=True)),
            "count": count,
        }
        for occurrence, count in sorted(expected_warmup_by_arm["capture"].items())
    ]
    drain_by_arm = {}
    for arm in ARMS:
        drain_by_arm[arm] = {
            **receipts[arm]["drain"],
            "phase_drains": receipts[arm]["phase_drains"],
        }
    drain = {
        **{key: sum(drain_by_arm[arm][key] for arm in ARMS) for key in DRAIN_FIELDS},
        "by_arm": drain_by_arm,
    }

    return {
        "schema_version": 2,
        "loaded_sessions": loaded_sessions,
        "profiling_root_admissions": frozen_admissions,
        "observed_profiling_root_admissions": {
            arm: aiperf_evidence[arm]["profiling_root_admissions"] for arm in ARMS
        },
        "events": events,
        "warmup_events": warmup_events,
        "expected_occurrences": expected_occurrences,
        "expected_warmup_occurrences": expected_warmup_occurrences,
        "expected_warmup_request_ids": {
            arm: sorted(warmup_specs_by_arm[arm]) for arm in ARMS
        },
        "drain": drain,
        "provenance": {
            "schema_version": 1,
            "reference_sha256": expected_reference_hash,
            "frozen_plan": plan_provenance,
            "runner_receipts": receipt_provenance,
            "request_logs": {
                arm: [
                    {"path": str(path), "sha256": digest}
                    for path, digest in resolved_logs[arm]
                ]
                for arm in ARMS
            },
            "execution_config": {
                "path": str(execution_config_path),
                "sha256": execution_config_hash,
            },
            "aiperf_artifacts": {
                arm: aiperf_evidence[arm]["artifacts"] for arm in ARMS
            },
            "request_log_partitions": {
                arm: {
                    key: {
                        "path": value["paths"][key],
                        "sha256": value[key]["sha256"],
                    }
                    for key in ("raw", "business", "control_sidecar")
                }
                for arm, value in partition_evidence.items()
            },
        },
    }
