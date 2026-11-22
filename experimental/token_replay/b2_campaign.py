from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time
from typing import Any, Callable


MAX_BUDGET_SECONDS = 14400
FROZEN_OCCURRENCE_COUNT = 53
DEFAULT_TEARDOWN_RESERVE_SECONDS = 300
GPU_INDEX = 0
GPU_MAX_USED_MIB = 500
ARMS = ("generated", "replay")
PAIR_ORDERS = (
    ("generated", "replay"),
    ("replay", "generated"),
    ("generated", "replay"),
    ("replay", "generated"),
    ("generated", "replay"),
)
ADJUSTED_T_CRITICAL = {3: 6.205347, 5: 3.495405}
DRAIN_FIELDS = ("sent", "completed", "errors", "skips", "cancellations", "pending")
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
RESERVED_ENV = {
    "CUDA_VISIBLE_DEVICES",
    "B2_CAMPAIGN_ID",
    "B2_CAMPAIGN_OUTPUT_DIR",
    "B2_PHASE",
    "B2_PHASE_ID",
    "B2_ARM",
    "B2_PAIR_ORDER_JSON",
    "B2_RESULT_PATH",
    "B2_STAGE_DIR",
    "B2_EXPECTED_OCCURRENCE_IDS_JSON",
    "B2_EXPECTED_OCCURRENCES_JSON",
    "B2_SELECTED_SESSIONS_JSON",
    "B2_ROOT_ADMISSIONS_JSON",
    "B2_CORRECTNESS_GATE_PATH",
    "B2_CORRECTNESS_GATE_SHA256",
    "B2_CAPTURE_REFERENCE_PATH",
    "B2_CAPTURE_REFERENCE_SHA256",
    "B2_CAMPAIGN_STATUS",
}


class CampaignError(ValueError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read {label} JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise CampaignError(f"{label} must be a JSON object")
    return value


def _artifact(path: Path, base: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve().relative_to(base.resolve())),
        "sha256": _sha256(path),
    }


def _positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignError(f"{label} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise CampaignError(f"{label} must be a positive finite number")
    return result


def _validate_command(value: Any, label: str, plan_base: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CampaignError(f"{label} must be an object")
    argv = value.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(arg, str) or not arg for arg in argv)
    ):
        raise CampaignError(f"{label}.argv must be a non-empty list of strings")
    timeout = _positive_number(value.get("timeout_seconds"), f"{label}.timeout_seconds")
    env = value.get("env", {})
    if not isinstance(env, dict) or any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in env.items()
    ):
        raise CampaignError(f"{label}.env must map strings to strings")
    reserved = RESERVED_ENV & set(env)
    if reserved:
        raise CampaignError(
            f"{label}.env overrides controller-owned keys: {sorted(reserved)}"
        )
    raw_cwd = value.get("cwd", ".")
    if not isinstance(raw_cwd, str) or not raw_cwd:
        raise CampaignError(f"{label}.cwd must be a non-empty path")
    cwd = Path(raw_cwd)
    if not cwd.is_absolute():
        cwd = plan_base / cwd
    cwd = cwd.resolve()
    if not cwd.is_dir():
        raise CampaignError(f"{label}.cwd is not a directory: {cwd}")
    return {"argv": argv, "timeout_seconds": timeout, "env": env, "cwd": cwd}


def _artifact_path(value: Any, base: Path, label: str) -> tuple[Path, str]:
    if not isinstance(value, dict):
        raise CampaignError(f"{label} must contain path and sha256")
    raw_path, expected_hash = value.get("path"), value.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise CampaignError(f"{label}.path must be a non-empty path")
    if not isinstance(expected_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_hash
    ):
        raise CampaignError(f"{label}.sha256 must be a lowercase SHA-256")
    path = Path(raw_path)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file() or _sha256(path) != expected_hash:
        raise CampaignError(f"{label} artifact is missing or its SHA-256 differs")
    return path, expected_hash


def _identity_key(value: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(value[key] for key in OCCURRENCE_FIELDS)


def _validate_gate_reference(
    value: Any,
    plan_base: Path,
    *,
    selected_sessions: list[str],
    root_admissions: list[dict[str, str]],
    occurrence_manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CampaignError("correctness_gate must contain path and sha256")
    path, expected_hash = _artifact_path(value, plan_base, "correctness_gate")
    gate = _read_json(path, "correctness_gate")
    if gate.get("status") != "PASS":
        raise CampaignError("correctness_gate status must be PASS")
    if (
        gate.get("evidence_scope") != "runtime"
        or gate.get("runtime_artifacts_verified") is not True
    ):
        raise CampaignError("correctness_gate must attest verified runtime artifacts")
    if gate.get("selected_sessions") != selected_sessions:
        raise CampaignError(
            "correctness_gate selected_sessions differ from the frozen cohort"
        )
    if gate.get("profiling_root_admissions") != root_admissions:
        raise CampaignError(
            "correctness_gate profiling root admissions differ from the frozen plan"
        )

    expected_count = len(occurrence_manifest)
    if gate.get("expected_occurrence_multiplicity") != expected_count:
        raise CampaignError(
            "correctness_gate occurrence multiplicity differs from the frozen plan"
        )
    if gate.get("event_counts") != {
        "capture": expected_count,
        "generated": expected_count,
        "replay": expected_count,
    }:
        raise CampaignError(
            "correctness_gate event counts differ from the frozen occurrence cohort"
        )

    provenance = gate.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("schema_version") != 1:
        raise CampaignError("correctness_gate has no runtime ledger provenance")
    frozen_plan_path, _ = _artifact_path(
        provenance.get("frozen_plan"),
        path.parent,
        "correctness_gate provenance.frozen_plan",
    )
    frozen_plan = _read_json(frozen_plan_path, "correctness_gate frozen plan")
    if frozen_plan.get("selected_sessions") != selected_sessions:
        raise CampaignError(
            "runtime frozen plan selected_sessions differ from the campaign cohort"
        )
    requests_by_arm = frozen_plan.get("requests_by_arm")
    expected_arms = {"capture", "generated", "replay"}
    if not isinstance(requests_by_arm, dict) or set(requests_by_arm) != expected_arms:
        raise CampaignError(
            "runtime frozen plan must contain capture, generated, and replay ledgers"
        )
    expected_identities = {_identity_key(item) for item in occurrence_manifest}
    for arm, requests in requests_by_arm.items():
        if not isinstance(requests, list) or len(requests) != expected_count:
            raise CampaignError(
                f"runtime frozen plan {arm} count differs from the frozen cohort"
            )
        identities: list[tuple[Any, ...]] = []
        request_ids: list[str] = []
        for item in requests:
            if not isinstance(item, dict) or not set(OCCURRENCE_FIELDS).issubset(item):
                raise CampaignError(
                    f"runtime frozen plan {arm} has an incomplete occurrence identity"
                )
            if not isinstance(item.get("request_id"), str) or not item["request_id"]:
                raise CampaignError(
                    f"runtime frozen plan {arm} has an occurrence without request_id"
                )
            identity = {key: item[key] for key in OCCURRENCE_FIELDS}
            if (
                not isinstance(identity["source_file"], str)
                or identity["source_file"] not in selected_sessions
                or identity["phase"] != "profiling"
                or any(
                    not isinstance(identity[key], str) or not identity[key]
                    for key in (
                        "phase",
                        "logical_root_id",
                        "tree_occurrence_id",
                        "cache_salt_class",
                    )
                )
                or any(
                    type(identity[key]) is not int or identity[key] < 0
                    for key in ("source_row", "turn", "reset_ordinal")
                )
            ):
                raise CampaignError(
                    f"runtime frozen plan {arm} has an invalid occurrence identity"
                )
            identities.append(_identity_key(item))
            request_ids.append(item["request_id"])
        if len(set(request_ids)) != len(request_ids):
            raise CampaignError(f"runtime frozen plan {arm} repeats request IDs")
        if (
            len(set(identities)) != len(identities)
            or set(identities) != expected_identities
        ):
            raise CampaignError(
                f"runtime frozen plan {arm} identities differ from the campaign cohort"
            )

    receipts = provenance.get("runner_receipts")
    logs = provenance.get("request_logs")
    if not isinstance(receipts, dict) or set(receipts) != expected_arms:
        raise CampaignError(
            "correctness_gate provenance must hash each arm runner receipt"
        )
    if not isinstance(logs, dict) or set(logs) != expected_arms:
        raise CampaignError(
            "correctness_gate provenance must hash each arm request log"
        )
    runtime_gate_gpu_seconds = 0.0
    for arm in ("capture", "generated", "replay"):
        receipt_path, _ = _artifact_path(
            receipts[arm],
            path.parent,
            f"correctness_gate provenance.runner_receipts.{arm}",
        )
        receipt = _read_json(receipt_path, f"correctness_gate {arm} runner receipt")
        active_seconds = receipt.get("gpu_active_seconds")
        if (
            receipt.get("schema_version") != 1
            or receipt.get("arm") != arm
            or isinstance(active_seconds, bool)
            or not isinstance(active_seconds, (int, float))
            or not math.isfinite(active_seconds)
            or active_seconds <= 0
        ):
            raise CampaignError(
                f"correctness_gate {arm} runner receipt is invalid or has invalid gpu_active_seconds"
            )
        runtime_gate_gpu_seconds += float(active_seconds)
        if not isinstance(logs[arm], list) or not logs[arm]:
            raise CampaignError(
                f"correctness_gate provenance.request_logs.{arm} must be non-empty"
            )
        for index, artifact in enumerate(logs[arm]):
            _artifact_path(
                artifact,
                path.parent,
                f"correctness_gate provenance.request_logs.{arm}[{index}]",
            )
    if not math.isfinite(runtime_gate_gpu_seconds) or runtime_gate_gpu_seconds <= 0:
        raise CampaignError(
            "correctness_gate runner receipts have invalid total gpu_active_seconds"
        )
    return {
        "path": path,
        "sha256": expected_hash,
        "result": gate,
        "runtime_gate_gpu_seconds": runtime_gate_gpu_seconds,
    }


def validate_plan(plan: Any, plan_base: Path) -> dict[str, Any]:
    if not isinstance(plan, dict) or plan.get("schema_version") != 1:
        raise CampaignError("plan schema_version must be 1")
    campaign_id = plan.get("campaign_id")
    if not isinstance(campaign_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", campaign_id
    ):
        raise CampaignError("campaign_id must be a 1-64 character path-safe identifier")

    budget = _positive_number(plan.get("hard_budget_seconds"), "hard_budget_seconds")
    if budget > MAX_BUDGET_SECONDS:
        raise CampaignError(
            f"hard_budget_seconds cannot exceed the {MAX_BUDGET_SECONDS}-second cap"
        )
    reserve = _positive_number(
        plan.get("teardown_reserve_seconds", DEFAULT_TEARDOWN_RESERVE_SECONDS),
        "teardown_reserve_seconds",
    )
    if reserve != DEFAULT_TEARDOWN_RESERVE_SECONDS:
        raise CampaignError(
            "teardown_reserve_seconds must equal the frozen "
            f"{DEFAULT_TEARDOWN_RESERVE_SECONDS}"
        )
    if reserve >= budget:
        raise CampaignError(
            "teardown_reserve_seconds must be less than hard_budget_seconds"
        )

    gpu = plan.get("gpu")
    if not isinstance(gpu, dict) or gpu.get("physical_index") != GPU_INDEX:
        raise CampaignError("B2 campaign is restricted to physical GPU 0")
    if gpu.get("max_used_mib") != GPU_MAX_USED_MIB:
        raise CampaignError(f"gpu.max_used_mib must be {GPU_MAX_USED_MIB}")

    sessions = plan.get("selected_sessions")
    if (
        not isinstance(sessions, list)
        or len(sessions) != 4
        or any(not isinstance(item, str) or not item for item in sessions)
        or len(set(sessions)) != 4
    ):
        raise CampaignError(
            "selected_sessions must contain four distinct session names"
        )
    roots = plan.get("profiling_root_admissions")
    if not isinstance(roots, list) or len(roots) != 4:
        raise CampaignError("profiling_root_admissions must contain exactly four roots")
    root_keys: set[tuple[str, str]] = set()
    for item in roots:
        if not isinstance(item, dict):
            raise CampaignError("each profiling root admission must be an object")
        session, root_id = item.get("source_file"), item.get("logical_root_id")
        if session not in sessions or not isinstance(root_id, str) or not root_id:
            raise CampaignError(
                "profiling root admission has invalid session/root identity"
            )
        root_keys.add((session, root_id))
    if len(root_keys) != 4:
        raise CampaignError(
            "profiling_root_admissions must contain four distinct roots"
        )

    occurrence_ids = plan.get("occurrence_ids")
    if (
        not isinstance(occurrence_ids, list)
        or len(occurrence_ids) != FROZEN_OCCURRENCE_COUNT
        or any(not isinstance(item, str) or not item for item in occurrence_ids)
        or len(occurrence_ids) != len(set(occurrence_ids))
    ):
        raise CampaignError(
            f"occurrence_ids must contain exactly {FROZEN_OCCURRENCE_COUNT} unique frozen IDs"
        )

    raw_manifest = plan.get("occurrence_manifest")
    if (
        not isinstance(raw_manifest, list)
        or len(raw_manifest) != FROZEN_OCCURRENCE_COUNT
    ):
        raise CampaignError(
            f"occurrence_manifest must contain exactly {FROZEN_OCCURRENCE_COUNT} identities"
        )
    occurrence_manifest: list[dict[str, Any]] = []
    manifest_ids: list[str] = []
    manifest_keys: list[tuple[Any, ...]] = []
    for index, item in enumerate(raw_manifest):
        if not isinstance(item, dict) or not set(
            OCCURRENCE_FIELDS + ("occurrence_id",)
        ).issubset(item):
            raise CampaignError(
                f"occurrence_manifest[{index}] is missing occurrence identity fields"
            )
        occurrence_id = item["occurrence_id"]
        if not isinstance(occurrence_id, str) or not occurrence_id:
            raise CampaignError(
                f"occurrence_manifest[{index}].occurrence_id must be non-empty"
            )
        identity = {key: item[key] for key in OCCURRENCE_FIELDS}
        if (
            not isinstance(identity["source_file"], str)
            or identity["source_file"] not in sessions
            or type(identity["source_row"]) is not int
            or identity["source_row"] < 0
            or identity["phase"] != "profiling"
            or any(
                not isinstance(identity[key], str) or not identity[key]
                for key in (
                    "phase",
                    "logical_root_id",
                    "tree_occurrence_id",
                    "cache_salt_class",
                )
            )
            or any(
                type(identity[key]) is not int or identity[key] < 0
                for key in ("turn", "reset_ordinal")
            )
        ):
            raise CampaignError(
                f"occurrence_manifest[{index}] has an invalid profiling identity"
            )
        manifest_ids.append(occurrence_id)
        manifest_keys.append(_identity_key(identity))
        occurrence_manifest.append({"occurrence_id": occurrence_id, **identity})
    if manifest_ids != occurrence_ids:
        raise CampaignError(
            "occurrence_manifest IDs differ from occurrence_ids in order"
        )
    if len(set(manifest_keys)) != FROZEN_OCCURRENCE_COUNT:
        raise CampaignError("occurrence_manifest contains duplicate frozen identities")
    manifest_roots = {
        (item["source_file"], item["logical_root_id"]) for item in occurrence_manifest
    }
    if manifest_roots != root_keys:
        raise CampaignError(
            "occurrence_manifest roots differ from profiling_root_admissions"
        )

    gate = _validate_gate_reference(
        plan.get("correctness_gate"),
        plan_base,
        selected_sessions=sessions,
        root_admissions=roots,
        occurrence_manifest=occurrence_manifest,
    )
    setup = _validate_command(plan.get("setup"), "setup", plan_base)
    teardown = _validate_command(plan.get("teardown"), "teardown", plan_base)
    if teardown["timeout_seconds"] > reserve:
        raise CampaignError(
            "teardown.timeout_seconds must fit inside the reserved teardown window"
        )

    phases_raw = plan.get("phases")
    if not isinstance(phases_raw, dict) or set(phases_raw) != {
        "capture",
        "warmup",
        "calibration_pair",
    }:
        raise CampaignError(
            "phases must define exactly capture, warmup, and calibration_pair"
        )
    phases: dict[str, Any] = {
        name: _validate_command(phases_raw[name], f"phases.{name}", plan_base)
        for name in ("capture", "warmup")
    }
    calibration = phases_raw["calibration_pair"]
    if not isinstance(calibration, dict) or set(calibration) != set(ARMS):
        raise CampaignError(
            "phases.calibration_pair must define generated and replay commands"
        )
    phases["calibration_pair"] = {
        arm: _validate_command(
            calibration[arm], f"phases.calibration_pair.{arm}", plan_base
        )
        for arm in ARMS
    }

    raw_pairs = plan.get("measurement_pairs")
    if not isinstance(raw_pairs, list) or len(raw_pairs) != 5:
        raise CampaignError(
            "measurement_pairs must predefine exactly five pair specifications"
        )
    pairs = []
    for index, item in enumerate(raw_pairs):
        pair_id = f"pair-{index + 1:02d}"
        if not isinstance(item, dict) or item.get("pair_id") != pair_id:
            raise CampaignError(
                f"measurement_pairs[{index}] must have pair_id {pair_id}"
            )
        if item.get("order") != list(PAIR_ORDERS[index]):
            raise CampaignError(
                f"{pair_id}.order differs from the frozen alternating sequence"
            )
        commands = item.get("commands")
        if not isinstance(commands, dict) or set(commands) != set(ARMS):
            raise CampaignError(f"{pair_id}.commands must define generated and replay")
        pairs.append(
            {
                "pair_id": pair_id,
                "order": list(PAIR_ORDERS[index]),
                "commands": {
                    arm: _validate_command(
                        commands[arm], f"{pair_id}.commands.{arm}", plan_base
                    )
                    for arm in ARMS
                },
            }
        )

    return {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "hard_budget_seconds": budget,
        "teardown_reserve_seconds": reserve,
        "gpu": gpu,
        "selected_sessions": sessions,
        "profiling_root_admissions": roots,
        "occurrence_ids": occurrence_ids,
        "occurrence_manifest": occurrence_manifest,
        "correctness_gate": gate,
        "setup": setup,
        "teardown": teardown,
        "phases": phases,
        "measurement_pairs": pairs,
    }


def parse_gpu0_memory(probe_output: str) -> int:
    if not isinstance(probe_output, str):
        raise CampaignError("GPU probe output must be text")
    matches = []
    for line in probe_output.splitlines():
        match = re.fullmatch(r"\s*(\d+)\s*,\s*(\d+)(?:\s*MiB)?\s*", line)
        if match:
            matches.append((int(match.group(1)), int(match.group(2))))
    gpu0 = [used for index, used in matches if index == GPU_INDEX]
    if len(gpu0) != 1:
        raise CampaignError("fresh GPU probe must report physical GPU 0 exactly once")
    return gpu0[0]


def _probe_gpu0() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise CampaignError(f"fresh GPU0 preflight failed: {error}") from error
    return result.stdout


def _run_argv(
    argv: list[str], cwd: Path, env: dict[str, str], timeout_seconds: float
) -> dict[str, Any]:
    started = time.monotonic()
    process: subprocess.Popen[bytes] | None = None
    timed_out = False
    error: str | None = None
    stdout = b""
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            stdout, _ = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                stdout, _ = process.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, _ = process.communicate()
    except OSError as exception:
        error = repr(exception)
    elapsed = time.monotonic() - started
    return {
        "returncode": process.returncode if process is not None else None,
        "timed_out": timed_out,
        "elapsed_seconds": elapsed,
        "error": error,
        "stdout": stdout,
    }


def _validate_drain(value: Any, label: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise CampaignError(f"{label}.drain must be an object")
    drain = {key: value.get(key) for key in DRAIN_FIELDS}
    if any(type(item) is not int or item < 0 for item in drain.values()):
        raise CampaignError(f"{label}.drain has missing or invalid counts")
    if drain["sent"] != drain["completed"] or any(
        drain[key] != 0 for key in ("errors", "skips", "cancellations", "pending")
    ):
        raise CampaignError(f"{label} did not drain cleanly")
    return drain


def _normalize_occurrences(
    value: Any,
    expected_manifest: list[dict[str, Any]],
    label: str,
    require_work: bool,
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise CampaignError(f"{label}.occurrences must be a list")
    result: dict[str, dict[str, Any]] = {}
    expected_by_id = {item["occurrence_id"]: item for item in expected_manifest}
    for item in value:
        if not isinstance(item, dict):
            raise CampaignError(f"{label} contains a non-object occurrence")
        occurrence_id = item.get("occurrence_id")
        request_id = item.get("request_id")
        if not isinstance(occurrence_id, str) or occurrence_id in result:
            raise CampaignError(f"{label} has a missing or duplicate occurrence_id")
        if not isinstance(request_id, str) or not request_id:
            raise CampaignError(f"{label}/{occurrence_id} has no request_id")
        if require_work:
            expected = expected_by_id.get(occurrence_id)
            if expected is None or any(
                item.get(key) != expected[key] for key in OCCURRENCE_FIELDS
            ):
                raise CampaignError(
                    f"{label}/{occurrence_id} identity differs from the frozen manifest"
                )
            output_ids = item.get("output_ids")
            if (
                not isinstance(output_ids, list)
                or not output_ids
                or any(type(token) is not int for token in output_ids)
            ):
                raise CampaignError(f"{label}/{occurrence_id} has invalid output_ids")
            output_length = item.get("output_length")
            if type(output_length) is not int or output_length != len(output_ids):
                raise CampaignError(
                    f"{label}/{occurrence_id} output_length differs from output_ids"
                )
            finish_reason = item.get("finish_reason")
            if not isinstance(finish_reason, str) or not finish_reason:
                raise CampaignError(f"{label}/{occurrence_id} has no finish_reason")
            for field in ("cache_hit_tokens", "new_prefill_tokens"):
                if type(item.get(field)) is not int or item[field] < 0:
                    raise CampaignError(f"{label}/{occurrence_id} has invalid {field}")
        result[occurrence_id] = item
    if set(result) != set(expected_by_id) or len(result) != len(expected_by_id):
        raise CampaignError(
            f"{label} occurrence ledger differs from the frozen occurrence IDs"
        )
    if len({item["request_id"] for item in result.values()}) != len(result):
        raise CampaignError(f"{label} has duplicate request IDs")
    return result


def _validate_stage_result(
    result: Any,
    *,
    phase: str,
    phase_id: str,
    arm: str,
    plan: dict[str, Any],
    require_work: bool,
    require_measurement: bool,
) -> dict[str, Any]:
    label = f"{phase_id}/{arm}"
    if not isinstance(result, dict) or result.get("schema_version") != 1:
        raise CampaignError(f"{label} result schema_version must be 1")
    if (
        result.get("status") != "PASS"
        or result.get("phase") != phase
        or result.get("arm") != arm
    ):
        raise CampaignError(
            f"{label} result must attest PASS with matching phase and arm"
        )
    if not isinstance(result.get("run_id"), str) or not result["run_id"]:
        raise CampaignError(f"{label} result has no run_id")
    if result.get("loaded_sessions") != plan["selected_sessions"]:
        raise CampaignError(f"{label} loaded_sessions differ from the frozen cohort")
    expected_roots = plan["profiling_root_admissions"] if require_work else []
    if result.get("profiling_root_admissions") != expected_roots:
        raise CampaignError(
            f"{label} profiling root admissions are invalid for this phase"
        )
    drain = _validate_drain(result.get("drain"), label)
    if drain["sent"] != drain["completed"]:
        raise CampaignError(f"{label} drain counts differ")
    occurrences = _normalize_occurrences(
        result.get("occurrences", []),
        plan["occurrence_manifest"] if require_work else [],
        label,
        require_work,
    )
    measurement = None
    if require_measurement:
        measurement = _positive_number(
            result.get("measurement_seconds"), f"{label}.measurement_seconds"
        )
    return {
        "run_id": result["run_id"],
        "loaded_sessions": result["loaded_sessions"],
        "profiling_root_admissions": result.get("profiling_root_admissions"),
        "drain": drain,
        "occurrences": occurrences,
        "measurement_seconds": measurement,
        "raw": result,
    }


def _same_outputs(
    actual: dict[str, dict[str, Any]], reference: dict[str, dict[str, Any]], label: str
) -> None:
    for occurrence_id, item in actual.items():
        expected = reference[occurrence_id]
        for field in ("output_ids", "output_length", "finish_reason"):
            if item[field] != expected[field]:
                raise CampaignError(
                    f"{label}/{occurrence_id} differs from capture on {field}"
                )


def _validate_pair(
    generated: dict[str, Any],
    replay: dict[str, Any],
    capture: dict[str, Any],
    label: str,
) -> float:
    gen_records = generated["occurrences"]
    replay_records = replay["occurrences"]
    _same_outputs(gen_records, capture["occurrences"], f"{label}/generated")
    _same_outputs(replay_records, capture["occurrences"], f"{label}/replay")
    for occurrence_id in gen_records:
        for field in ("output_ids", "output_length", "finish_reason"):
            if (
                gen_records[occurrence_id][field]
                != replay_records[occurrence_id][field]
            ):
                raise CampaignError(
                    f"{label}/{occurrence_id} generated/replay {field} differs"
                )
        for field in ("cache_hit_tokens", "new_prefill_tokens"):
            if (
                gen_records[occurrence_id][field]
                != replay_records[occurrence_id][field]
            ):
                raise CampaignError(
                    f"{label}/{occurrence_id} generated/replay {field} differs"
                )
    gen_seconds = generated["measurement_seconds"]
    replay_seconds = replay["measurement_seconds"]
    if not isinstance(gen_seconds, (int, float)) or not isinstance(
        replay_seconds, (int, float)
    ):
        raise CampaignError(f"{label} is missing paired measurement durations")
    return replay_seconds / gen_seconds - 1.0


def paired_interval(values: list[float]) -> dict[str, Any]:
    if len(values) not in ADJUSTED_T_CRITICAL or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in values
    ):
        raise CampaignError(
            "paired_interval requires three or five finite pair measurements"
        )
    mean = statistics.mean(values)
    half = (
        ADJUSTED_T_CRITICAL[len(values)]
        * statistics.stdev(values)
        / math.sqrt(len(values))
    )
    low, high = mean - half, mean + half
    touches_target = low <= 0.03 <= high
    if len(values) == 3 and touches_target:
        decision = "EXPAND_TO_FIVE"
    elif high <= 0.03:
        decision = "PASS"
    elif low > 0.03:
        decision = "FAIL"
    else:
        decision = "INCONCLUSIVE"
    return {
        "pair_count": len(values),
        "mean_overhead_fraction": mean,
        "ci95": [low, high],
        "adjusted_student_t_critical": ADJUSTED_T_CRITICAL[len(values)],
        "threshold_fraction": 0.03,
        "touches_threshold": touches_target,
        "decision": decision,
    }


class _CampaignRunner:
    def __init__(
        self,
        plan: dict[str, Any],
        output_dir: Path,
        *,
        gpu_probe: Callable[[], str],
        monotonic: Callable[[], float],
    ) -> None:
        self.plan = plan
        self.output_dir = output_dir
        self.gpu_probe = gpu_probe
        self.monotonic = monotonic
        self.runtime_gate_gpu_seconds = plan["correctness_gate"][
            "runtime_gate_gpu_seconds"
        ]
        self.start_time: float | None = None
        self.deadline: float | None = None
        self.stage_receipts: list[dict[str, Any]] = []
        self.run_ids: set[str] = set()
        self.capture: dict[str, Any] | None = None
        self.capture_result_path: Path | None = None
        self.capture_result_sha256: str | None = None
        self.status = "INCOMPLETE"
        self.reason: str | None = None
        self.pair_receipts: list[dict[str, Any]] = []

    def _remaining(self) -> float:
        if self.deadline is None:
            return 0.0
        return self.deadline - self.monotonic()

    def _can_start(self, commands: list[dict[str, Any]]) -> bool:
        reserve = self.plan["teardown_reserve_seconds"]
        required = sum(command["timeout_seconds"] for command in commands)
        return self._remaining() >= required + reserve

    def _environment(
        self,
        command: dict[str, Any],
        *,
        phase: str,
        phase_id: str,
        arm: str,
        stage_dir: Path,
        result_path: Path,
        pair_order: list[str] | None = None,
    ) -> dict[str, str]:
        env = os.environ.copy()
        env.update(command["env"])
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "B2_CAMPAIGN_ID": self.plan["campaign_id"],
                "B2_CAMPAIGN_OUTPUT_DIR": str(self.output_dir.resolve()),
                "B2_PHASE": phase,
                "B2_PHASE_ID": phase_id,
                "B2_ARM": arm,
                "B2_PAIR_ORDER_JSON": json.dumps(pair_order or []),
                "B2_RESULT_PATH": str(result_path.resolve()),
                "B2_STAGE_DIR": str(stage_dir.resolve()),
                "B2_EXPECTED_OCCURRENCE_IDS_JSON": json.dumps(
                    self.plan["occurrence_ids"]
                ),
                "B2_EXPECTED_OCCURRENCES_JSON": json.dumps(
                    self.plan["occurrence_manifest"]
                ),
                "B2_SELECTED_SESSIONS_JSON": json.dumps(self.plan["selected_sessions"]),
                "B2_ROOT_ADMISSIONS_JSON": json.dumps(
                    self.plan["profiling_root_admissions"]
                ),
                "B2_CORRECTNESS_GATE_PATH": str(self.plan["correctness_gate"]["path"]),
                "B2_CORRECTNESS_GATE_SHA256": self.plan["correctness_gate"]["sha256"],
                "B2_CAPTURE_REFERENCE_PATH": (
                    str(self.capture_result_path.resolve())
                    if self.capture_result_path
                    else ""
                ),
                "B2_CAPTURE_REFERENCE_SHA256": self.capture_result_sha256 or "",
                "B2_CAMPAIGN_STATUS": self.status,
            }
        )
        return env

    def _write_not_started(self, stage_id: str, reason: str, arm: str = "") -> None:
        stage_dir = self.output_dir / stage_id
        stage_dir.mkdir(parents=True, exist_ok=False)
        receipt = {
            "schema_version": 1,
            "stage_id": stage_id,
            "arm": arm,
            "status": "NOT_STARTED",
            "reason": reason,
            "correctness_gate": _artifact(
                self.plan["correctness_gate"]["path"], self.output_dir
            )
            if self.plan["correctness_gate"]["path"].is_relative_to(self.output_dir)
            else {
                "path": str(self.plan["correctness_gate"]["path"]),
                "sha256": self.plan["correctness_gate"]["sha256"],
            },
        }
        _write_json(stage_dir / "receipt.json", receipt)
        self.stage_receipts.append(
            {"path": str(stage_dir / "receipt.json"), "status": "NOT_STARTED"}
        )

    def _run_stage(
        self,
        *,
        stage_id: str,
        phase: str,
        arm: str,
        command: dict[str, Any],
        require_work: bool,
        require_measurement: bool,
        pair_order: list[str] | None = None,
    ) -> dict[str, Any]:
        stage_dir = self.output_dir / stage_id
        stage_dir.mkdir(parents=True, exist_ok=False)
        result_path = stage_dir / "result.json"
        started = self.monotonic()
        env = self._environment(
            command,
            phase=phase,
            phase_id=stage_id,
            arm=arm,
            stage_dir=stage_dir,
            result_path=result_path,
            pair_order=pair_order,
        )
        execution = _run_argv(
            command["argv"],
            command["cwd"],
            env,
            command["timeout_seconds"],
        )
        stdout = execution.pop("stdout")
        log_path = stage_dir / "command.log"
        log_path.write_bytes(stdout)
        receipt: dict[str, Any] = {
            "schema_version": 1,
            "campaign_id": self.plan["campaign_id"],
            "stage_id": stage_id,
            "phase": phase,
            "arm": arm,
            "argv": command["argv"],
            "cwd": str(command["cwd"]),
            "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
            "started_elapsed_seconds": started - (self.start_time or started),
            **execution,
            "command_log": {"path": log_path.name, "sha256": _sha256(log_path)},
            "correctness_gate": {
                "path": str(self.plan["correctness_gate"]["path"]),
                "sha256": self.plan["correctness_gate"]["sha256"],
            },
            "status": "FAILED",
        }
        try:
            if execution["error"]:
                raise CampaignError(
                    f"{stage_id} failed to execute: {execution['error']}"
                )
            if execution["timed_out"]:
                raise CampaignError(f"{stage_id} exceeded its command timeout")
            if execution["returncode"] != 0:
                raise CampaignError(
                    f"{stage_id} command exited with status {execution['returncode']}"
                )
            if not result_path.is_file():
                raise CampaignError(f"{stage_id} did not write B2_RESULT_PATH")
            raw_result = _read_json(result_path, f"{stage_id} result")
            validated = _validate_stage_result(
                raw_result,
                phase=phase,
                phase_id=stage_id,
                arm=arm,
                plan=self.plan,
                require_work=require_work,
                require_measurement=require_measurement,
            )
            if validated["run_id"] in self.run_ids:
                raise CampaignError(f"{stage_id} repeats run_id {validated['run_id']}")
            self.run_ids.add(validated["run_id"])
            receipt.update(
                {
                    "status": "PASS",
                    "run_id": validated["run_id"],
                    "drain": validated["drain"],
                    "result": _artifact(result_path, stage_dir),
                    "occurrence_count": len(validated["occurrences"]),
                    "measurement_seconds": validated["measurement_seconds"],
                }
            )
            _write_json(stage_dir / "receipt.json", receipt)
            self.stage_receipts.append(
                {
                    "path": str(stage_dir / "receipt.json"),
                    "sha256": _sha256(stage_dir / "receipt.json"),
                    "status": "PASS",
                }
            )
            return validated
        except CampaignError as error:
            receipt["error"] = str(error)
            _write_json(stage_dir / "receipt.json", receipt)
            self.stage_receipts.append(
                {
                    "path": str(stage_dir / "receipt.json"),
                    "sha256": _sha256(stage_dir / "receipt.json"),
                    "status": "FAILED",
                }
            )
            raise

    def _run_setup(self) -> dict[str, Any]:
        command = self.plan["setup"]
        stage_dir = self.output_dir / "setup"
        stage_dir.mkdir(parents=True, exist_ok=False)
        started = self.monotonic()
        env = os.environ.copy()
        env.update(command["env"])
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "B2_CAMPAIGN_ID": self.plan["campaign_id"],
                "B2_CAMPAIGN_OUTPUT_DIR": str(self.output_dir.resolve()),
                "B2_CAMPAIGN_STATUS": self.status,
            }
        )
        execution = _run_argv(
            command["argv"], command["cwd"], env, command["timeout_seconds"]
        )
        log_path = stage_dir / "command.log"
        log_path.write_bytes(execution.pop("stdout"))
        status = (
            "PASS"
            if (
                execution["returncode"] == 0
                and not execution["timed_out"]
                and not execution["error"]
            )
            else "FAILED"
        )
        receipt = {
            "schema_version": 1,
            "campaign_id": self.plan["campaign_id"],
            "stage_id": "setup",
            "phase": "setup",
            "arm": "server",
            "argv": command["argv"],
            "cwd": str(command["cwd"]),
            "cuda_visible_devices": "0",
            "started_elapsed_seconds": started - (self.start_time or started),
            **execution,
            "command_log": {"path": log_path.name, "sha256": _sha256(log_path)},
            "correctness_gate": {
                "path": str(self.plan["correctness_gate"]["path"]),
                "sha256": self.plan["correctness_gate"]["sha256"],
            },
            "status": status,
        }
        _write_json(stage_dir / "receipt.json", receipt)
        self.stage_receipts.append(
            {
                "path": str(stage_dir / "receipt.json"),
                "sha256": _sha256(stage_dir / "receipt.json"),
                "status": status,
            }
        )
        if status != "PASS":
            raise CampaignError("setup command failed or timed out")
        return receipt

    def _run_pair(
        self,
        pair: dict[str, Any],
        *,
        phase: str,
        pair_kind: str,
        capture: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        pair_id = pair["pair_id"]
        order = pair["order"]
        commands = [pair["commands"][arm] for arm in order]
        if not self._can_start(commands):
            self._write_not_started(
                f"{pair_id}-generated",
                "teardown reserve would be consumed",
                "generated",
            )
            self._write_not_started(
                f"{pair_id}-replay", "teardown reserve would be consumed", "replay"
            )
            raise CampaignError(f"insufficient remaining budget to start {pair_id}")
        results: dict[str, dict[str, Any]] = {}
        try:
            for arm in order:
                results[arm] = self._run_stage(
                    stage_id=f"{pair_id}-{arm}",
                    phase=phase,
                    arm=arm,
                    command=pair["commands"][arm],
                    require_work=True,
                    require_measurement=True,
                    pair_order=order,
                )
            overhead = _validate_pair(
                results["generated"], results["replay"], capture, pair_id
            )
            pair_receipt = {
                "schema_version": 1,
                "pair_id": pair_id,
                "phase": phase,
                "order": order,
                "correctness_gate": {
                    "path": str(self.plan["correctness_gate"]["path"]),
                    "sha256": self.plan["correctness_gate"]["sha256"],
                },
                "capture_reference": {
                    "path": str(self.capture_result_path),
                    "sha256": self.capture_result_sha256,
                },
                "arms": {
                    arm: {
                        "stage_receipt": f"{pair_id}-{arm}/receipt.json",
                        "stage_receipt_sha256": _sha256(
                            self.output_dir / f"{pair_id}-{arm}" / "receipt.json"
                        ),
                        "result": _artifact(
                            self.output_dir / f"{pair_id}-{arm}" / "result.json",
                            self.output_dir,
                        ),
                        "measurement_seconds": results[arm]["measurement_seconds"],
                    }
                    for arm in ARMS
                },
                "pair_data": {
                    "matched_occurrence_count": len(self.plan["occurrence_ids"]),
                    "occurrence_ids": self.plan["occurrence_ids"],
                    "output_and_cache_work_matched": True,
                    "overhead_fraction": overhead,
                },
                "status": "PASS",
            }
            pair_dir = self.output_dir / pair_id
            pair_dir.mkdir(parents=True, exist_ok=False)
            _write_json(pair_dir / "receipt.json", pair_receipt)
            pair_receipt["receipt_sha256"] = _sha256(pair_dir / "receipt.json")
            self.pair_receipts.append(pair_receipt)
            return overhead, pair_receipt
        except CampaignError:
            pair_dir = self.output_dir / pair_id
            pair_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                pair_dir / "receipt.json",
                {
                    "schema_version": 1,
                    "pair_id": pair_id,
                    "phase": phase,
                    "order": order,
                    "status": "FAILED",
                    "arms_completed": [arm for arm in order if arm in results],
                    "correctness_gate": {
                        "path": str(self.plan["correctness_gate"]["path"]),
                        "sha256": self.plan["correctness_gate"]["sha256"],
                    },
                },
            )
            raise

    def _run_single_phase(
        self, phase: str, command: dict[str, Any], arm: str
    ) -> dict[str, Any]:
        if not self._can_start([command]):
            self._write_not_started(phase, "teardown reserve would be consumed", arm)
            raise CampaignError(f"insufficient remaining budget to start {phase}")
        return self._run_stage(
            stage_id=phase,
            phase=phase,
            arm=arm,
            command=command,
            require_work=(phase == "capture"),
            require_measurement=False,
        )

    def _write_pair_skips(self, start_index: int, reason: str) -> None:
        for pair in self.plan["measurement_pairs"][start_index:]:
            pair_id = pair["pair_id"]
            for arm in ARMS:
                if not (self.output_dir / f"{pair_id}-{arm}").exists():
                    self._write_not_started(f"{pair_id}-{arm}", reason, arm)
            pair_dir = self.output_dir / pair_id
            pair_dir.mkdir(parents=True, exist_ok=True)
            if not (pair_dir / "receipt.json").exists():
                _write_json(
                    pair_dir / "receipt.json",
                    {
                        "schema_version": 1,
                        "pair_id": pair_id,
                        "phase": "measurement",
                        "order": pair["order"],
                        "status": "NOT_STARTED",
                        "reason": reason,
                    },
                )

    def _fill_unstarted_pairs(self, reason: str) -> None:
        for index, pair in enumerate(self.plan["measurement_pairs"]):
            pair_id = pair["pair_id"]
            pair_dir = self.output_dir / pair_id
            if (pair_dir / "receipt.json").exists():
                continue
            for arm in ARMS:
                if not (self.output_dir / f"{pair_id}-{arm}").exists():
                    self._write_not_started(f"{pair_id}-{arm}", reason, arm)
            pair_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                pair_dir / "receipt.json",
                {
                    "schema_version": 1,
                    "pair_id": pair_id,
                    "phase": "measurement",
                    "order": pair["order"],
                    "status": "NOT_STARTED",
                    "reason": reason,
                },
            )

    def execute(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=False)
        _write_json(self.output_dir / "plan.json", self.plan["raw_plan"])
        self.start_time = self.monotonic() - self.runtime_gate_gpu_seconds
        self.deadline = self.start_time + self.plan["hard_budget_seconds"]
        try:
            gpu_output = self.gpu_probe()
            used_mib = parse_gpu0_memory(gpu_output)
            if used_mib >= GPU_MAX_USED_MIB:
                self.status = "BLOCKED_GPU_GUARD"
                self.reason = (
                    f"physical GPU 0 has {used_mib} MiB used (limit {GPU_MAX_USED_MIB})"
                )
                return self._finish_without_setup(used_mib)
        except CampaignError as error:
            self.status = "BLOCKED_GPU_GUARD"
            self.reason = str(error)
            return self._finish_without_setup(None)

        setup_attempted = False
        setup_receipt: dict[str, Any] | None = None
        teardown_receipt: dict[str, Any] | None = None
        estimates: list[float] = []
        final_interval: dict[str, Any] | None = None
        try:
            if not self._can_start([self.plan["setup"]]):
                raise CampaignError("budget reserve prevents setup")
            setup_attempted = True
            setup_receipt = self._run_setup()
            self._run_single_phase("capture", self.plan["phases"]["capture"], "capture")
            capture_path = self.output_dir / "capture" / "result.json"
            capture_result = _read_json(capture_path, "capture result")
            self.capture = _validate_stage_result(
                capture_result,
                phase="capture",
                phase_id="capture",
                arm="capture",
                plan=self.plan,
                require_work=True,
                require_measurement=False,
            )
            self.capture_result_path = capture_path
            self.capture_result_sha256 = _sha256(capture_path)

            self._run_single_phase("warmup", self.plan["phases"]["warmup"], "warmup")

            calibration = {
                "pair_id": "calibration-pair",
                "order": ["generated", "replay"],
                "commands": self.plan["phases"]["calibration_pair"],
            }
            self._run_pair(
                calibration,
                phase="calibration",
                pair_kind="calibration",
                capture=self.capture,
            )

            for index, pair in enumerate(self.plan["measurement_pairs"][:3]):
                estimate, _ = self._run_pair(
                    pair,
                    phase="measurement",
                    pair_kind="measurement",
                    capture=self.capture,
                )
                estimates.append(estimate)
            interval = paired_interval(estimates)
            final_interval = interval
            if interval["decision"] == "EXPAND_TO_FIVE":
                for pair in self.plan["measurement_pairs"][3:5]:
                    estimate, _ = self._run_pair(
                        pair,
                        phase="measurement",
                        pair_kind="measurement",
                        capture=self.capture,
                    )
                    estimates.append(estimate)
                    if len(estimates) == 5:
                        final_interval = paired_interval(estimates)
            else:
                self._write_pair_skips(
                    3, "three-pair interval did not touch the 3% threshold"
                )

            if final_interval is None:
                raise CampaignError("paired estimate was not produced")
            if final_interval["decision"] == "PASS":
                self.status = "PASS"
            elif final_interval["decision"] == "FAIL":
                self.status = "FAIL_OVERHEAD_THRESHOLD"
            else:
                self.status = "INCONCLUSIVE"
            if len(estimates) == 3 and final_interval["decision"] == "EXPAND_TO_FIVE":
                self.status = "INCOMPLETE"
                self.reason = "three-pair interval touched the threshold but budget blocked expansion"
        except CampaignError as error:
            self.status = "INCOMPLETE"
            self.reason = str(error)
            self._fill_unstarted_pairs(str(error))
            if len(estimates) == 3:
                try:
                    final_interval = paired_interval(estimates)
                except CampaignError:
                    final_interval = None
            elif len(estimates) == 4:
                final_interval = None
        finally:
            if setup_attempted:
                teardown_receipt = self._run_teardown()
            else:
                teardown_receipt = self._not_run_teardown("setup was not attempted")

        if teardown_receipt.get("status") != "PASS" and self.status in {
            "PASS",
            "FAIL_OVERHEAD_THRESHOLD",
            "INCONCLUSIVE",
        }:
            self.status = "INCOMPLETE"
            self.reason = "teardown failed after experiment phases completed"
        elapsed = self.monotonic() - (
            self.start_time if self.start_time is not None else self.monotonic()
        )
        campaign_receipt = {
            "schema_version": 1,
            "campaign_id": self.plan["campaign_id"],
            "status": self.status,
            "reason": self.reason,
            "gpu": {"physical_index": GPU_INDEX, "used_mib_preflight": used_mib},
            "budget": {
                "hard_budget_seconds": self.plan["hard_budget_seconds"],
                "teardown_reserve_seconds": self.plan["teardown_reserve_seconds"],
                "runtime_gate_gpu_seconds": self.runtime_gate_gpu_seconds,
                "elapsed_seconds_including_gate_and_teardown": elapsed,
                "within_hard_budget": elapsed <= self.plan["hard_budget_seconds"],
            },
            "correctness_gate": {
                "path": str(self.plan["correctness_gate"]["path"]),
                "sha256": self.plan["correctness_gate"]["sha256"],
            },
            "capture_reference": (
                {
                    "path": str(self.capture_result_path),
                    "sha256": self.capture_result_sha256,
                }
                if self.capture_result_path
                else None
            ),
            "measurement_overheads": estimates,
            "paired_interval": final_interval,
            "paired_interval_status": (
                "COMPLETE"
                if final_interval is not None
                else "INCOMPLETE_FIVE_PAIR_EXPANSION"
                if len(estimates) == 4
                else "UNAVAILABLE"
            ),
            "stages": self.stage_receipts,
            "pairs": self.pair_receipts,
            "setup": setup_receipt,
            "teardown": teardown_receipt,
        }
        _write_json(self.output_dir / "campaign-receipt.json", campaign_receipt)
        if elapsed > self.plan["hard_budget_seconds"]:
            campaign_receipt["status"] = "INCOMPLETE"
            campaign_receipt["reason"] = "hard campaign budget exceeded"
            _write_json(self.output_dir / "campaign-receipt.json", campaign_receipt)
        return campaign_receipt

    def _run_teardown(self) -> dict[str, Any]:
        teardown_dir = self.output_dir / "teardown"
        teardown_dir.mkdir(parents=True, exist_ok=False)
        remaining = self._remaining()
        if remaining <= 0:
            receipt = {
                "schema_version": 1,
                "phase": "teardown",
                "status": "FAILED",
                "reason": "hard budget expired before teardown command could start",
                "elapsed_seconds": 0.0,
            }
            _write_json(teardown_dir / "receipt.json", receipt)
            return receipt
        timeout = min(self.plan["teardown"]["timeout_seconds"], remaining)
        command = self.plan["teardown"]
        env = os.environ.copy()
        env.update(command["env"])
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "B2_CAMPAIGN_ID": self.plan["campaign_id"],
                "B2_CAMPAIGN_OUTPUT_DIR": str(self.output_dir.resolve()),
                "B2_CAMPAIGN_STATUS": self.status,
            }
        )
        execution = _run_argv(command["argv"], command["cwd"], env, timeout)
        log_path = teardown_dir / "command.log"
        log_path.write_bytes(execution.pop("stdout"))
        status = (
            "PASS"
            if (
                execution["returncode"] == 0
                and not execution["timed_out"]
                and not execution["error"]
            )
            else "FAILED"
        )
        receipt = {
            "schema_version": 1,
            "phase": "teardown",
            "status": status,
            "argv": command["argv"],
            "cuda_visible_devices": "0",
            "timeout_seconds": timeout,
            **execution,
            "command_log": {"path": log_path.name, "sha256": _sha256(log_path)},
        }
        _write_json(teardown_dir / "receipt.json", receipt)
        return receipt

    def _not_run_teardown(self, reason: str) -> dict[str, Any]:
        teardown_dir = self.output_dir / "teardown"
        teardown_dir.mkdir(parents=True, exist_ok=False)
        receipt = {
            "schema_version": 1,
            "phase": "teardown",
            "status": "NOT_RUN",
            "reason": reason,
        }
        _write_json(teardown_dir / "receipt.json", receipt)
        return receipt

    def _finish_without_setup(self, used_mib: int | None) -> dict[str, Any]:
        teardown = self._not_run_teardown(
            "GPU preflight did not pass; no server was started"
        )
        elapsed = self.monotonic() - (
            self.start_time if self.start_time is not None else self.monotonic()
        )
        receipt = {
            "schema_version": 1,
            "campaign_id": self.plan["campaign_id"],
            "status": self.status,
            "reason": self.reason,
            "gpu": {"physical_index": GPU_INDEX, "used_mib_preflight": used_mib},
            "budget": {
                "hard_budget_seconds": self.plan["hard_budget_seconds"],
                "teardown_reserve_seconds": self.plan["teardown_reserve_seconds"],
                "runtime_gate_gpu_seconds": self.runtime_gate_gpu_seconds,
                "elapsed_seconds_including_gate_and_teardown": elapsed,
                "within_hard_budget": elapsed <= self.plan["hard_budget_seconds"],
            },
            "correctness_gate": {
                "path": str(self.plan["correctness_gate"]["path"]),
                "sha256": self.plan["correctness_gate"]["sha256"],
            },
            "stages": [],
            "pairs": [],
            "teardown": teardown,
        }
        _write_json(self.output_dir / "campaign-receipt.json", receipt)
        return receipt


def run_campaign(
    plan_path: Path,
    output_dir: Path,
    *,
    execute: bool = False,
    gpu_probe: Callable[[], str] | None = None,
    monotonic: Callable[[], float] | None = None,
) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise CampaignError(
            f"output directory already exists; refusing overwrite: {output_dir}"
        )
    raw_plan = _read_json(plan_path, "campaign plan")
    plan = validate_plan(raw_plan, plan_path.parent)
    plan["raw_plan"] = raw_plan
    if not execute:
        return {
            "schema_version": 1,
            "campaign_id": plan["campaign_id"],
            "status": "VALIDATED_NO_EXECUTION",
            "arms_launched": False,
            "output_dir": str(output_dir),
            "hard_budget_seconds": plan["hard_budget_seconds"],
        }
    runner = _CampaignRunner(
        plan,
        output_dir,
        gpu_probe=gpu_probe or _probe_gpu0,
        monotonic=monotonic or time.monotonic,
    )
    return runner.execute()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate or run a frozen B2 AgentX campaign. Execution requires --execute "
            "and hashed runtime-gate timing receipts."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute the plan's explicit argv commands; omitted means validate only",
    )
    args = parser.parse_args()
    try:
        result = run_campaign(
            args.plan,
            args.output_dir,
            execute=args.execute,
        )
    except CampaignError as error:
        print(
            json.dumps({"status": "INVALID_PLAN", "error": str(error)}), file=sys.stderr
        )
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] in {"VALIDATED_NO_EXECUTION", "PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
