from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

from b2_aiperf_adapter import AIPerfAdapterError, create_observation
from b2_cpu_gate import ARMS, DRAIN_FIELDS, OCCURRENCE_FIELDS, select_cohort
from b2_sglang_ledger import (
    RuntimeLedgerError,
    build_event_bundle,
    materialize_capture_reference,
)


class RuntimeRunnerError(ValueError):
    pass


SERVER_LOGGING = {
    "enabled": True,
    "format": "json",
    "level": 3,
    "incremental_streaming_output": False,
    "request_id_header_logged": True,
    "log_exceeded_ms": -1,
    "dedicated_target": True,
}
RESERVED_ENV = {
    "AGENTX_B2_ARM",
    "AGENTX_TOKEN_TRACE",
    "AGENTX_NUM_SESSIONS",
    "AGENTX_REPLAY_MODE",
    "SGLANG_LOG_REQUEST_HEADERS",
    "SGLANG_LOG_REQUEST_EXCEEDED_MS",
    "B2_DATASET_DIR",
    "B2_REQUEST_LOG",
    "B2_AIPERF_ARTIFACT_DIR",
    "B2_OBSERVATION",
    "B2_EXPECTED_REQUEST_IDS_JSON",
    "B2_EXPECTED_WARMUP_REQUEST_IDS_JSON",
    "B2_EXPECTED_WARMUP_OCCURRENCES_JSON",
    "B2_SELECTED_SESSIONS_JSON",
    "B2_ROOT_ADMISSIONS_JSON",
}
PROFILE_ROOT_ADMISSION_COUNT = 4


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeRunnerError(f"cannot read JSON {path}: {error}") from error


def _nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _identity(occurrence: dict[str, Any]) -> dict[str, Any]:
    return {key: occurrence[key] for key in OCCURRENCE_FIELDS}


def _request_id(arm: str, occurrence: dict[str, Any]) -> str:
    digest = _sha256_bytes(
        _canonical_bytes({"arm": arm, "occurrence": _identity(occurrence)})
    )
    return f"b2-{arm}-{digest[:24]}"


def _phase_request_id(arm: str, occurrence: dict[str, Any], phase: str) -> str:
    profile_occurrence = {**occurrence, "phase": "profiling"}
    return f"{_request_id(arm, profile_occurrence)}-{phase}"


def _resolve_path(raw: Any, base: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise RuntimeRunnerError(f"{label} must be a non-empty path")
    path = Path(raw)
    return (path if path.is_absolute() else base / path).resolve()


def _safe_child(root: Path, relative: str, label: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise RuntimeRunnerError(
            f"{label} must be a relative path inside the output directory"
        )
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise RuntimeRunnerError(f"{label} escapes the output directory")
    return resolved


def _validate_occurrences(
    raw: Any,
    selected_sessions: list[str],
    reference_outputs: dict[tuple[str, int], tuple[int, ...]],
    expected_phase: str = "profiling",
    allow_empty: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or (not raw and not allow_empty):
        raise RuntimeRunnerError("occurrences must be a non-empty frozen list")
    selected = set(selected_sessions)
    seen_identity: set[tuple[Any, ...]] = set()
    seen_locator: set[tuple[str, int, int | None]] = set()
    occurrences: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise RuntimeRunnerError(f"occurrences[{index}] must be an object")
        if not set(OCCURRENCE_FIELDS).issubset(item):
            raise RuntimeRunnerError(
                f"occurrences[{index}] is missing occurrence identity fields"
            )
        identity = _identity(item)
        for key in (
            "source_file",
            "phase",
            "logical_root_id",
            "tree_occurrence_id",
            "cache_salt_class",
        ):
            if not isinstance(identity[key], str) or not identity[key]:
                raise RuntimeRunnerError(
                    f"occurrences[{index}].{key} must be a non-empty string"
                )
        for key in ("source_row", "turn", "reset_ordinal"):
            if not _nonnegative_int(identity[key]):
                raise RuntimeRunnerError(
                    f"occurrences[{index}].{key} must be a non-negative integer"
                )
        if identity["phase"] != expected_phase:
            raise RuntimeRunnerError(
                f"occurrences[{index}].phase must be {expected_phase!r}"
            )
        if identity["source_file"] not in selected:
            raise RuntimeRunnerError(
                f"occurrences[{index}] references an unloaded session"
            )
        if (identity["source_file"], identity["source_row"]) not in reference_outputs:
            raise RuntimeRunnerError(
                f"occurrences[{index}] is outside the immutable reference"
            )
        identity_key = tuple(identity[key] for key in OCCURRENCE_FIELDS)
        if identity_key in seen_identity:
            raise RuntimeRunnerError(
                f"duplicate frozen occurrence identity at index {index}"
            )
        seen_identity.add(identity_key)

        locator = item.get("locator")
        if not isinstance(locator, dict):
            raise RuntimeRunnerError(f"occurrences[{index}].locator must be an object")
        outer_index = locator.get("outer_index")
        inner_index = locator.get("inner_index")
        if not _nonnegative_int(outer_index) or not (
            inner_index is None or _nonnegative_int(inner_index)
        ):
            raise RuntimeRunnerError(
                f"occurrences[{index}].locator has invalid source indices"
            )
        locator_key = (identity["source_file"], outer_index, inner_index)
        if locator_key in seen_locator:
            raise RuntimeRunnerError(f"duplicate Weka request locator at index {index}")
        seen_locator.add(locator_key)
        occurrences.append({**identity, "locator": locator})
    return occurrences


def _all_weka_locators(
    trace: dict[str, Any], source_file: str
) -> set[tuple[str, int, int | None]]:
    requests = trace.get("requests")
    if not isinstance(requests, list) or not requests:
        raise RuntimeRunnerError(f"{source_file} has no Weka requests")
    locators: set[tuple[str, int, int | None]] = set()
    for outer_index, request in enumerate(requests):
        if not isinstance(request, dict):
            raise RuntimeRunnerError(
                f"{source_file} request {outer_index} is not an object"
            )
        request_type = request.get("type")
        if request_type == "subagent":
            inner_requests = request.get("requests")
            if not isinstance(inner_requests, list):
                raise RuntimeRunnerError(
                    f"{source_file} subagent {outer_index} has no requests list"
                )
            for inner_index, inner in enumerate(inner_requests):
                if not isinstance(inner, dict) or inner.get("type") not in {"n", "s"}:
                    raise RuntimeRunnerError(
                        f"{source_file} subagent {outer_index} request {inner_index} is invalid"
                    )
                locators.add((source_file, outer_index, inner_index))
        elif request_type in {"n", "s"}:
            locators.add((source_file, outer_index, None))
        else:
            raise RuntimeRunnerError(
                f"{source_file} request {outer_index} has unsupported type"
            )
    return locators


def _locate_weka_request(
    trace: dict[str, Any], locator: dict[str, Any], label: str
) -> dict[str, Any]:
    outer_index = locator["outer_index"]
    try:
        request = trace["requests"][outer_index]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeRunnerError(
            f"{label} locator outer_index is out of range"
        ) from error
    inner_index = locator["inner_index"]
    if inner_index is None:
        if request.get("type") not in {"n", "s"}:
            raise RuntimeRunnerError(
                f"{label} locator points to a non-request Weka entry"
            )
        return request
    if request.get("type") != "subagent":
        raise RuntimeRunnerError(f"{label} inner_index requires a subagent Weka entry")
    try:
        nested = request["requests"][inner_index]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeRunnerError(
            f"{label} locator inner_index is out of range"
        ) from error
    if not isinstance(nested, dict) or nested.get("type") not in {"n", "s"}:
        raise RuntimeRunnerError(
            f"{label} locator points to a non-request subagent entry"
        )
    return nested


def _validate_roots(
    raw: Any, selected_sessions: list[str], occurrences: list[dict[str, Any]]
) -> list[dict[str, str]]:
    if not isinstance(raw, list) or len(raw) != PROFILE_ROOT_ADMISSION_COUNT:
        raise RuntimeRunnerError(
            "profiling_root_admissions must contain exactly four roots"
        )
    selected = set(selected_sessions)
    roots: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise RuntimeRunnerError(
                f"profiling_root_admissions[{index}] must be an object"
            )
        source_file = item.get("source_file")
        root_id = item.get("logical_root_id")
        if not isinstance(source_file, str) or source_file not in selected:
            raise RuntimeRunnerError(
                f"profiling_root_admissions[{index}] has an invalid source_file"
            )
        if not isinstance(root_id, str) or not root_id:
            raise RuntimeRunnerError(
                f"profiling_root_admissions[{index}] has an invalid logical_root_id"
            )
        key = (source_file, root_id)
        if key in seen:
            raise RuntimeRunnerError(
                "profiling_root_admissions contains a duplicate root"
            )
        seen.add(key)
        roots.append({"source_file": source_file, "logical_root_id": root_id})
    event_roots = {
        (item["source_file"], item["logical_root_id"])
        for item in occurrences
        if item["phase"].lower() == "profiling"
    }
    if event_roots != seen:
        raise RuntimeRunnerError(
            "frozen profiling root admissions must equal roots represented by profiling occurrences"
        )
    return roots


def _stage_datasets(
    source_dir: Path,
    output_dir: Path,
    selected_sessions: list[str],
    occurrences: list[dict[str, Any]],
    warmup_occurrences: list[dict[str, Any]],
    reference_outputs: dict[tuple[str, int], tuple[int, ...]],
    validate_output_vectors: bool = True,
) -> dict[str, Path]:
    source_traces: dict[str, dict[str, Any]] = {}
    actual_locators: set[tuple[str, int, int | None]] = set()
    for session in selected_sessions:
        if Path(session).name != session or session in {".", ".."}:
            raise RuntimeRunnerError(f"invalid Weka session filename: {session!r}")
        trace = _read_json(source_dir / session)
        if not isinstance(trace, dict):
            raise RuntimeRunnerError(f"Weka trace {session} must be a JSON object")
        source_traces[session] = trace
        actual_locators |= _all_weka_locators(trace, session)

    occurrence_locators = {
        (
            item["source_file"],
            item["locator"]["outer_index"],
            item["locator"]["inner_index"],
        )
        for item in occurrences
    }
    if actual_locators != occurrence_locators:
        missing = sorted(actual_locators - occurrence_locators)
        extra = sorted(occurrence_locators - actual_locators)
        raise RuntimeRunnerError(
            f"frozen occurrence locators must cover every loaded Weka request; "
            f"unmapped={missing[:3]}, invalid={extra[:3]}"
        )

    for item in [*occurrences, *warmup_occurrences]:
        locator_key = (
            item["source_file"],
            item["locator"]["outer_index"],
            item["locator"]["inner_index"],
        )
        if locator_key not in actual_locators:
            raise RuntimeRunnerError(
                f"frozen {item['phase']} occurrence locator is outside the loaded source traces"
            )
        trace = source_traces[item["source_file"]]
        request = _locate_weka_request(trace, item["locator"], item["source_file"])
        expected = reference_outputs[(item["source_file"], item["source_row"])]
        output_ids = request.get("output_ids")
        if not isinstance(output_ids, list) or (
            tuple(output_ids) != expected
            if validate_output_vectors
            else len(output_ids) != len(expected)
        ):
            raise RuntimeRunnerError(
                f"Weka source output differs from immutable reference for "
                f"{item['source_file']} row {item['source_row']}"
            )
        if not isinstance(request.get("input_ids"), list) or not request["input_ids"]:
            raise RuntimeRunnerError(
                f"Weka source is missing exact input_ids for {item['source_file']} row {item['source_row']}"
            )

    dataset_dirs: dict[str, Path] = {}
    for arm in ARMS:
        arm_dataset = output_dir / arm / "dataset"
        arm_dataset.mkdir(parents=True, exist_ok=False)
        traces = json.loads(json.dumps(source_traces))
        for item in occurrences:
            trace = traces[item["source_file"]]
            request = _locate_weka_request(trace, item["locator"], item["source_file"])
            request["request_id"] = _request_id(arm, item)
            request["occurrence_identity"] = _identity(item)
        for session in selected_sessions:
            _write_json(arm_dataset / session, traces[session])
        dataset_dirs[arm] = arm_dataset
    return dataset_dirs


def _patch_dataset_outputs(
    dataset_dir: Path,
    occurrences: list[dict[str, Any]],
    reference_outputs: dict[tuple[str, int], tuple[int, ...]],
) -> None:
    traces: dict[str, dict[str, Any]] = {}
    for item in occurrences:
        session = item["source_file"]
        if session not in traces:
            value = _read_json(dataset_dir / session)
            if not isinstance(value, dict):
                raise RuntimeRunnerError(
                    f"staged Weka trace {session} must be a JSON object"
                )
            traces[session] = value
        request = _locate_weka_request(traces[session], item["locator"], session)
        key = (session, item["source_row"])
        output_ids = reference_outputs.get(key)
        if output_ids is None:
            raise RuntimeRunnerError(
                f"capture reference has no output for {session} row {item['source_row']}"
            )
        request["output_ids"] = list(output_ids)
    for session, trace in traces.items():
        _write_json(dataset_dir / session, trace)


def _validate_arm_output_equality(bundle: dict[str, Any]) -> None:
    events = bundle.get("events")
    if not isinstance(events, list) or not events:
        raise RuntimeRunnerError(
            "runtime event bundle has no events for output comparison"
        )
    by_arm: dict[str, dict[tuple[Any, ...], tuple[int, ...]]] = {
        arm: {} for arm in ARMS
    }
    for event in events:
        arm = event.get("arm") if isinstance(event, dict) else None
        if arm not in by_arm:
            raise RuntimeRunnerError("runtime event bundle contains an unknown arm")
        key = tuple(event.get(field) for field in OCCURRENCE_FIELDS)
        output_ids = event.get("output_ids")
        if key in by_arm[arm]:
            raise RuntimeRunnerError(
                f"duplicate output event for {arm} occurrence {key}"
            )
        if not isinstance(output_ids, list) or any(
            type(token) is not int for token in output_ids
        ):
            raise RuntimeRunnerError(
                f"invalid output vector for {arm} occurrence {key}"
            )
        by_arm[arm][key] = tuple(output_ids)
    capture = by_arm["capture"]
    if not capture:
        raise RuntimeRunnerError("capture output ledger is empty")
    for arm in ("generated", "replay"):
        if set(by_arm[arm]) != set(capture):
            raise RuntimeRunnerError(
                f"{arm} output ledger identities differ from capture"
            )
        mismatches = [
            key for key, output in capture.items() if by_arm[arm][key] != output
        ]
        if mismatches:
            raise RuntimeRunnerError(
                f"{arm} output vectors differ from capture at {mismatches[0]}"
            )


def _build_frozen_plan(
    reference: dict[str, Any],
    selected_sessions: list[str],
    occurrences: list[dict[str, Any]],
    warmup_occurrences: list[dict[str, Any]],
    root_admissions: list[dict[str, str]],
) -> dict[str, Any]:
    requests_by_arm = {
        arm: [
            {
                "request_id": _phase_request_id(arm, item, "profiling"),
                **_identity(item),
            }
            for item in occurrences
        ]
        for arm in ARMS
    }
    warmup_requests_by_arm = {
        arm: [
            {
                "request_id": _phase_request_id(arm, item, "warmup"),
                **_identity(item),
                "locator": item["locator"],
            }
            for item in warmup_occurrences
        ]
        for arm in ARMS
    }
    return {
        "schema_version": 1,
        "reference_sha256": _sha256_bytes(_canonical_bytes(reference)),
        "selected_sessions": selected_sessions,
        "profiling_root_admissions": root_admissions,
        "requests_by_arm": requests_by_arm,
        "warmup_requests_by_arm": warmup_requests_by_arm,
    }


def _run_command(
    argv: list[str], cwd: Path, env: dict[str, str], log_path: Path, timeout: float
) -> None:
    if not argv or any(not isinstance(arg, str) or not arg for arg in argv):
        raise RuntimeRunnerError("arm command argv must be a non-empty list of strings")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log_file:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=0.25)
            except subprocess.TimeoutExpired:
                pass
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.05)
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
            raise RuntimeRunnerError(
                f"arm command timed out after {timeout:g}s; process group terminated"
            ) from error
    if return_code != 0:
        raise RuntimeRunnerError(
            f"arm command exited with status {return_code}; see {log_path}"
        )


def _validate_observation(
    arm: str,
    observation_path: Path,
    expected_profile_ids: list[str],
    expected_warmup_ids: list[str],
    selected_sessions: list[str],
    root_admissions: list[dict[str, str]],
    completed_from_logs: int,
) -> dict[str, Any]:
    observation = _read_json(observation_path)
    if not isinstance(observation, dict):
        raise RuntimeRunnerError(f"{arm} observation must be a JSON object")
    if observation.get("schema_version") != 1 or observation.get("arm") != arm:
        raise RuntimeRunnerError(f"{arm} observation has an invalid schema or arm")
    if not isinstance(observation.get("run_id"), str) or not observation["run_id"]:
        raise RuntimeRunnerError(f"{arm} observation has no run_id")
    if observation.get("loaded_sessions") != selected_sessions:
        raise RuntimeRunnerError(
            f"{arm} observation has an asymmetric loaded-session record"
        )
    if observation.get("profiling_root_admissions") != root_admissions:
        raise RuntimeRunnerError(
            f"{arm} observation has asymmetric profiling-root admissions"
        )
    skip_evidence = observation.get("skip_evidence")
    if (
        not isinstance(skip_evidence, dict)
        or skip_evidence.get("count") != 0
        or not isinstance(skip_evidence.get("per_request_artifact"), dict)
        or not isinstance(skip_evidence["per_request_artifact"].get("sha256"), str)
        or len(skip_evidence["per_request_artifact"]["sha256"]) != 64
    ):
        raise RuntimeRunnerError(
            f"{arm} observation has no independently checked zero-skip evidence"
        )
    request_ids = observation.get("request_ids")
    if (
        not isinstance(request_ids, list)
        or not all(isinstance(value, str) and value for value in request_ids)
        or len(request_ids) != len(set(request_ids))
        or set(request_ids) != set(expected_profile_ids)
    ):
        raise RuntimeRunnerError(
            f"{arm} observation profiling request IDs differ from the frozen plan"
        )
    warmup_request_ids = observation.get("warmup_request_ids")
    if (
        not isinstance(warmup_request_ids, list)
        or not all(isinstance(value, str) and value for value in warmup_request_ids)
        or len(warmup_request_ids) != len(set(warmup_request_ids))
        or set(warmup_request_ids) != set(expected_warmup_ids)
    ):
        raise RuntimeRunnerError(
            f"{arm} observation warmup request IDs differ from the frozen plan"
        )
    drain = {key: observation.get(key) for key in DRAIN_FIELDS}
    if any(not _nonnegative_int(value) for value in drain.values()):
        raise RuntimeRunnerError(
            f"{arm} observation has incomplete or invalid drain counts"
        )
    expected_total = len(expected_profile_ids) + len(expected_warmup_ids)
    if drain["sent"] != expected_total or drain["completed"] != completed_from_logs:
        raise RuntimeRunnerError(
            f"{arm} observation drain counts disagree with request logs"
        )
    if drain["sent"] != drain["completed"] or any(
        drain[key] != 0 for key in ("errors", "skips", "cancellations", "pending")
    ):
        raise RuntimeRunnerError(f"{arm} did not drain cleanly")
    phase_drains = observation.get("phase_drains")
    if not isinstance(phase_drains, dict) or set(phase_drains) != {
        "warmup",
        "profiling",
    }:
        raise RuntimeRunnerError(
            f"{arm} observation has no complete per-phase drain evidence"
        )
    for phase, expected_count in (
        ("warmup", len(expected_warmup_ids)),
        ("profiling", len(expected_profile_ids)),
    ):
        phase_drain = phase_drains[phase]
        if not isinstance(phase_drain, dict) or any(
            not _nonnegative_int(phase_drain.get(field))
            for field in ("sent", "completed", "errors", "cancellations", "pending")
        ):
            raise RuntimeRunnerError(
                f"{arm} observation has incomplete {phase} drain counts"
            )
        if (
            phase_drain["sent"] != expected_count
            or phase_drain["completed"] != expected_count
        ):
            raise RuntimeRunnerError(f"{arm} {phase} drain differs from frozen IDs")
    return observation


def _count_finished_events(log_path: Path, arm: str) -> int:
    if not log_path.is_file():
        raise RuntimeRunnerError(f"{arm} request log was not produced: {log_path}")
    count = 0
    for line_number, line in enumerate(
        log_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeRunnerError(
                f"invalid JSON in {log_path}:{line_number}"
            ) from error
        if not isinstance(record, dict):
            raise RuntimeRunnerError(f"non-object event in {log_path}:{line_number}")
        if record.get("event") == "request.finished":
            count += 1
    return count


def _artifact(path: Path, base: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve().relative_to(base.resolve())),
        "sha256": _sha256_bytes(path.read_bytes()),
    }


def run_b2_plan(
    plan_path: Path, output_dir: Path, prepare_only: bool = False
) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    plan_base = plan_path.parent
    plan = _read_json(plan_path)
    if not isinstance(plan, dict) or plan.get("schema_version") != 1:
        raise RuntimeRunnerError("run plan schema_version must be 1")
    if output_dir.exists():
        raise RuntimeRunnerError(
            f"output directory already exists; refusing overwrite: {output_dir}"
        )

    reference_path = _resolve_path(plan.get("reference"), plan_base, "reference")
    source_dir = _resolve_path(
        plan.get("source_sessions_dir"), plan_base, "source_sessions_dir"
    )
    reference = _read_json(reference_path)
    if not isinstance(reference, dict):
        raise RuntimeRunnerError("reference must be a JSON object")
    reference_mode = plan.get("reference_mode", "source")
    if reference_mode not in {"source", "capture_derived"}:
        raise RuntimeRunnerError("reference_mode must be 'source' or 'capture_derived'")
    cohort = select_cohort(reference)
    selected_sessions = cohort["selected_sessions"]
    if plan.get("selected_sessions") != selected_sessions:
        raise RuntimeRunnerError(
            "run plan selected_sessions differs from the seed-42 fixed cohort"
        )
    if not source_dir.is_dir():
        raise RuntimeRunnerError(f"source_sessions_dir does not exist: {source_dir}")

    reference_outputs: dict[tuple[str, int], tuple[int, ...]] = {}
    for row in reference.get("rows", []):
        session, request, output_ids = (
            row.get("session"),
            row.get("request"),
            row.get("output_ids"),
        )
        if (
            isinstance(session, str)
            and type(request) is int
            and isinstance(output_ids, list)
        ):
            reference_outputs[(session, request)] = tuple(output_ids)
    occurrences = _validate_occurrences(
        plan.get("occurrences"), selected_sessions, reference_outputs
    )
    warmup_occurrences = _validate_occurrences(
        plan.get("warmup_occurrences"),
        selected_sessions,
        reference_outputs,
        expected_phase="warmup",
        allow_empty=True,
    )
    root_admissions = _validate_roots(
        plan.get("profiling_root_admissions"), selected_sessions, occurrences
    )

    arms = plan.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise RuntimeRunnerError(
            "arms must contain exactly capture, generated, and replay"
        )
    validated_arms: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        spec = arms[arm]
        if not isinstance(spec, dict) or not isinstance(spec.get("argv"), list):
            raise RuntimeRunnerError(f"arms.{arm}.argv must be a list")
        if any(not isinstance(arg, str) or not arg for arg in spec["argv"]):
            raise RuntimeRunnerError(f"arms.{arm}.argv must contain non-empty strings")
        timeout = spec.get("timeout_seconds")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise RuntimeRunnerError(f"arms.{arm}.timeout_seconds must be positive")
        env_extra = spec.get("env", {})
        if not isinstance(env_extra, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in env_extra.items()
        ):
            raise RuntimeRunnerError(f"arms.{arm}.env must map strings to strings")
        reserved = RESERVED_ENV & set(env_extra)
        if reserved:
            raise RuntimeRunnerError(
                f"arms.{arm}.env overrides runner-owned keys: {sorted(reserved)}"
            )
        cwd = _resolve_path(spec.get("cwd", "."), plan_base, f"arms.{arm}.cwd")
        if not cwd.is_dir():
            raise RuntimeRunnerError(f"arms.{arm}.cwd is not a directory: {cwd}")
        validated_arms[arm] = {
            "argv": spec["argv"],
            "timeout_seconds": timeout,
            "env": env_extra,
            "cwd": cwd,
        }
        artifact_rel = spec.get("aiperf_artifact_dir")
        if not isinstance(artifact_rel, str) or not artifact_rel:
            raise RuntimeRunnerError(
                f"arms.{arm}.aiperf_artifact_dir must be an explicit relative path"
            )
        artifact_path = Path(artifact_rel)
        if artifact_path.is_absolute() or ".." in artifact_path.parts:
            raise RuntimeRunnerError(
                f"arms.{arm}.aiperf_artifact_dir must be relative to its arm output directory"
            )
        validated_arms[arm]["aiperf_artifact_dir"] = artifact_rel
        if spec.get("gpu_lifecycle_included") is not True:
            raise RuntimeRunnerError(
                f"arms.{arm}.gpu_lifecycle_included must be true for budget timing evidence"
            )
        validated_arms[arm]["gpu_lifecycle_included"] = True

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        dataset_dirs = _stage_datasets(
            source_dir,
            output_dir,
            selected_sessions,
            occurrences,
            warmup_occurrences,
            reference_outputs,
            validate_output_vectors=reference_mode == "source",
        )
        frozen_plan = _build_frozen_plan(
            reference,
            selected_sessions,
            occurrences,
            warmup_occurrences,
            root_admissions,
        )
        frozen_plan["reference_mode"] = reference_mode
        frozen_plan_path = output_dir / "frozen-plan.json"
        capture_reference = reference
        frozen_plan_hash: str | None = None
        capture_reference_path: Path | None = None
        if reference_mode == "source":
            _write_json(frozen_plan_path, frozen_plan)
            frozen_plan_hash = _sha256_bytes(frozen_plan_path.read_bytes())
        elif prepare_only:
            _write_json(frozen_plan_path, frozen_plan)
        execution_config_path = output_dir / "execution-config.json"
        _write_json(
            execution_config_path,
            {
                "schema_version": 1,
                "arms": {
                    arm: {
                        "argv": validated_arms[arm]["argv"],
                        "cwd": str(validated_arms[arm]["cwd"]),
                        "timeout_seconds": validated_arms[arm]["timeout_seconds"],
                        "env": validated_arms[arm]["env"],
                        "aiperf_artifact_dir": validated_arms[arm][
                            "aiperf_artifact_dir"
                        ],
                        "gpu_lifecycle_included": validated_arms[arm][
                            "gpu_lifecycle_included"
                        ],
                    }
                    for arm in ARMS
                },
            },
        )
        execution_config_hash = _sha256_bytes(execution_config_path.read_bytes())

        if prepare_only:
            status = {
                "schema_version": 1,
                "status": "VALIDATED_NO_ARMS_RUN",
                "evidence_scope": "plan-validation-only",
                "arms_executed": False,
                "selected_sessions": selected_sessions,
                "occurrence_count_per_arm": len(occurrences),
                "frozen_plan": frozen_plan_path.name,
                "execution_config": _artifact(execution_config_path, output_dir),
            }
            _write_json(output_dir / "status.json", status)
            return status

        request_logs: dict[str, list[dict[str, str]]] = {}
        runner_receipts: dict[str, dict[str, str]] = {}
        request_log_partitions: dict[str, dict[str, dict[str, str]]] = {}
        aiperf_artifacts: dict[str, dict[str, Any]] = {}
        arm_timings: dict[str, dict[str, Any]] = {}
        run_ids: set[str] = set()
        for arm in ARMS:
            arm_dir = output_dir / arm
            request_log = arm_dir / "server-request.jsonl"
            raw_request_log = arm_dir / "server-request-raw.jsonl"
            business_request_log = arm_dir / "server-request-business.jsonl"
            control_request_log = arm_dir / "server-request-control.jsonl"
            aiperf_artifact_dir = _safe_child(
                arm_dir,
                validated_arms[arm]["aiperf_artifact_dir"],
                f"arms.{arm}.aiperf_artifact_dir",
            )
            aiperf_artifact_dir.mkdir(parents=True, exist_ok=False)
            observation_path = arm_dir / "runner-observation.json"
            command_log = arm_dir / "command.log"
            request_ids = [
                _phase_request_id(arm, item, "profiling") for item in occurrences
            ]
            warmup_request_ids = [
                _phase_request_id(arm, item, "warmup") for item in warmup_occurrences
            ]
            arm_env = os.environ.copy()
            arm_env.update(validated_arms[arm]["env"])
            arm_env.update(
                {
                    "AGENTX_B2_ARM": arm,
                    "AGENTX_TOKEN_TRACE": str(dataset_dirs[arm]),
                    "AGENTX_NUM_SESSIONS": str(len(selected_sessions)),
                    "AGENTX_REPLAY_MODE": "recorded"
                    if arm == "replay"
                    else "generated",
                    "SGLANG_LOG_REQUEST_HEADERS": "x-request-id,x-b2-occurrence",
                    "SGLANG_LOG_REQUEST_EXCEEDED_MS": "-1",
                    "B2_DATASET_DIR": str(dataset_dirs[arm]),
                    "B2_REQUEST_LOG": str(request_log),
                    "B2_AIPERF_ARTIFACT_DIR": str(aiperf_artifact_dir),
                    "B2_OBSERVATION": str(observation_path),
                    "B2_EXPECTED_REQUEST_IDS_JSON": json.dumps(request_ids),
                    "B2_EXPECTED_WARMUP_REQUEST_IDS_JSON": json.dumps(
                        warmup_request_ids
                    ),
                    "B2_EXPECTED_WARMUP_OCCURRENCES_JSON": json.dumps(
                        frozen_plan["warmup_requests_by_arm"][arm]
                    ),
                    "B2_SELECTED_SESSIONS_JSON": json.dumps(selected_sessions),
                    "B2_ROOT_ADMISSIONS_JSON": json.dumps(root_admissions),
                }
            )
            wall_start = _utc_now()
            monotonic_start = time.monotonic()
            try:
                _run_command(
                    validated_arms[arm]["argv"],
                    validated_arms[arm]["cwd"],
                    arm_env,
                    command_log,
                    float(validated_arms[arm]["timeout_seconds"]),
                )
            except Exception:
                elapsed = time.monotonic() - monotonic_start
                arm_timings[arm] = {
                    "wall_start_utc": wall_start,
                    "wall_end_utc": _utc_now(),
                    "monotonic_start_seconds": monotonic_start,
                    "monotonic_end_seconds": monotonic_start + elapsed,
                    "elapsed_seconds": elapsed,
                    "gpu_active_seconds": elapsed,
                    "clock_semantics": "monotonic interval around arm wrapper process, including wrapper-managed server startup and teardown",
                }
                _write_json(
                    output_dir / "runtime-timing.partial.json",
                    {
                        "schema_version": 1,
                        "timing_scope": "sequential-arm-process-wall-time",
                        "clock_semantics": "monotonic duration around each arm wrapper process, including wrapper-managed server startup and teardown",
                        "arm_timings": arm_timings,
                        "gpu_active_seconds": sum(
                            item["gpu_active_seconds"] for item in arm_timings.values()
                        ),
                        "execution_config": _artifact(
                            execution_config_path, output_dir
                        ),
                        "completed_runner_receipts": runner_receipts,
                    },
                )
                raise
            elapsed = time.monotonic() - monotonic_start
            wall_end = _utc_now()
            arm_timings[arm] = {
                "wall_start_utc": wall_start,
                "wall_end_utc": wall_end,
                "monotonic_start_seconds": monotonic_start,
                "monotonic_end_seconds": monotonic_start + elapsed,
                "elapsed_seconds": elapsed,
                "gpu_active_seconds": elapsed,
                "clock_semantics": "monotonic interval around arm wrapper process, including wrapper-managed server startup and teardown",
            }
            profile_root_by_id = {
                _phase_request_id(arm, item, "profiling"): (
                    item["source_file"],
                    item["logical_root_id"],
                )
                for item in occurrences
            }
            create_observation(
                arm=arm,
                artifact_dir=aiperf_artifact_dir,
                client_log_path=command_log,
                server_log_source=request_log,
                raw_log_path=raw_request_log,
                business_log_path=business_request_log,
                control_log_path=control_request_log,
                observation_path=observation_path,
                expected_profile_ids=request_ids,
                expected_warmup_ids=warmup_request_ids,
                profile_root_by_id=profile_root_by_id,
                selected_sessions=selected_sessions,
                root_admissions=root_admissions,
            )
            if arm == "capture" and reference_mode == "capture_derived":
                capture_reference = materialize_capture_reference(
                    reference, frozen_plan, business_request_log
                )
                capture_reference_path = output_dir / "capture-reference.json"
                _write_json(capture_reference_path, capture_reference)
                capture_outputs = {
                    (row["session"], row["request"]): tuple(row["output_ids"])
                    for row in capture_reference["rows"]
                }
                _patch_dataset_outputs(
                    dataset_dirs["replay"], occurrences, capture_outputs
                )
                frozen_plan = _build_frozen_plan(
                    capture_reference,
                    selected_sessions,
                    occurrences,
                    warmup_occurrences,
                    root_admissions,
                )
                frozen_plan["reference_mode"] = reference_mode
                _write_json(frozen_plan_path, frozen_plan)
                frozen_plan_hash = _sha256_bytes(frozen_plan_path.read_bytes())
            if frozen_plan_hash is None:
                raise RuntimeRunnerError(
                    "frozen plan hash was not materialized before receipt creation"
                )
            completed_from_logs = _count_finished_events(business_request_log, arm)
            observation = _validate_observation(
                arm,
                observation_path,
                request_ids,
                warmup_request_ids,
                selected_sessions,
                root_admissions,
                completed_from_logs,
            )
            if observation["run_id"] in run_ids:
                raise RuntimeRunnerError(
                    "arm runner receipts must have distinct run_id values"
                )
            run_ids.add(observation["run_id"])
            receipt_path = arm_dir / "runner-receipt.json"
            receipt = {
                "schema_version": 1,
                "run_id": observation["run_id"],
                "arm": arm,
                "frozen_plan_sha256": frozen_plan_hash,
                "execution_config_sha256": execution_config_hash,
                "gpu_active_seconds": elapsed,
                "loaded_sessions": selected_sessions,
                "profiling_root_admissions": root_admissions,
                "request_ids": request_ids,
                "warmup_request_ids": warmup_request_ids,
                **{key: observation[key] for key in DRAIN_FIELDS},
                "phase_drains": observation["phase_drains"],
                "skip_evidence": observation["skip_evidence"],
                "aiperf_artifacts": observation["aiperf_artifacts"],
                "request_log_partition": observation["request_log_partition"],
                "arm_timing": arm_timings[arm],
            }
            _write_json(receipt_path, receipt)
            request_logs[arm] = [_artifact(business_request_log, output_dir)]
            runner_receipts[arm] = _artifact(receipt_path, output_dir)
            request_log_partitions[arm] = {
                key: observation["request_log_partition"][key]
                for key in ("raw", "business", "control_sidecar")
            }
            aiperf_artifacts[arm] = observation["aiperf_artifacts"]

        manifest = {
            "schema_version": 1,
            "server_logging": SERVER_LOGGING,
            "frozen_plan": _artifact(frozen_plan_path, output_dir),
            "request_logs": request_logs,
            "runner_receipts": runner_receipts,
            "request_log_partitions": request_log_partitions,
            "aiperf_artifacts": aiperf_artifacts,
            "execution_config": _artifact(execution_config_path, output_dir),
            "reference_mode": reference_mode,
        }
        if capture_reference_path is not None:
            manifest["capture_reference"] = _artifact(
                capture_reference_path, output_dir
            )
        manifest_path = output_dir / "runtime-manifest.json"
        _write_json(manifest_path, manifest)
        bundle = build_event_bundle(manifest, output_dir, capture_reference)
        if reference_mode == "capture_derived":
            _validate_arm_output_equality(bundle)
        bundle_path = output_dir / "event-bundle.json"
        _write_json(bundle_path, bundle)
        status = {
            "schema_version": 1,
            "status": "READY_FOR_CPU_GATE",
            "arms": list(ARMS),
            "selected_sessions": selected_sessions,
            "occurrence_count_per_arm": len(occurrences),
            "warmup_occurrence_count_per_arm": len(warmup_occurrences),
            "runtime_manifest": manifest_path.name,
            "event_bundle": bundle_path.name,
            "execution_config": _artifact(execution_config_path, output_dir),
        }
        _write_json(output_dir / "status.json", status)
        timing_path = output_dir / "runtime-timing.json"
        _write_json(
            timing_path,
            {
                "schema_version": 1,
                "timing_scope": "sequential-arm-process-wall-time",
                "clock_semantics": "monotonic duration around each arm wrapper process, including wrapper-managed server startup and teardown",
                "arm_timings": arm_timings,
                "gpu_active_seconds": sum(
                    item["gpu_active_seconds"] for item in arm_timings.values()
                ),
                "execution_config": _artifact(execution_config_path, output_dir),
                "output_hashes": {
                    "frozen_plan": _artifact(frozen_plan_path, output_dir),
                    "runtime_manifest": _artifact(manifest_path, output_dir),
                    "event_bundle": _artifact(bundle_path, output_dir),
                },
            },
        )
        status["timing_receipt"] = _artifact(timing_path, output_dir)
        _write_json(output_dir / "status.json", status)
        return status
    except (
        OSError,
        RuntimeRunnerError,
        RuntimeLedgerError,
        AIPerfAdapterError,
        json.JSONDecodeError,
    ) as error:
        _write_json(
            output_dir / "failure.json",
            {"schema_version": 1, "status": "INCOMPLETE", "error": str(error)},
        )
        raise RuntimeRunnerError(str(error)) from error


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run three symmetric B2 arms and emit a hashed runtime manifest."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="validate and stage the frozen inputs without launching any arm commands",
    )
    args = parser.parse_args()
    try:
        print(
            json.dumps(
                run_b2_plan(args.plan, args.output_dir, prepare_only=args.prepare_only),
                ensure_ascii=False,
            )
        )
    except RuntimeRunnerError as error:
        print(
            json.dumps(
                {"status": "INCOMPLETE", "error": str(error)}, ensure_ascii=False
            ),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
