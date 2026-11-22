from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from b2_sglang_ledger import RuntimeLedgerError, build_event_bundle


ARMS = ("capture", "generated", "replay")
PROFILE_ROOT_ADMISSION_COUNT = 4
REQUIRED_EVENT_FIELDS = frozenset(
    {
        "event",
        "arm",
        "request_id",
        "source_file",
        "source_row",
        "phase",
        "logical_root_id",
        "tree_occurrence_id",
        "turn",
        "reset_ordinal",
        "cache_salt_class",
        "output_ids",
        "output_length",
        "cache_hit_tokens",
        "new_prefill_tokens",
        "total_retractions",
    }
)
DRAIN_FIELDS = (
    "sent",
    "completed",
    "errors",
    "skips",
    "cancellations",
    "pending",
)
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


class GateInputError(ValueError):
    pass


def select_cohort(
    reference: dict[str, Any], seed: int = 42, size: int = 4
) -> dict[str, Any]:
    rows = reference.get("rows")
    if reference.get("complete") is not True or not isinstance(rows, list):
        raise GateInputError("corpus reference must be complete and contain rows")
    if type(reference.get("requests_exact")) is not int or reference[
        "requests_exact"
    ] != len(rows):
        raise GateInputError("corpus reference request count does not match its rows")

    sessions: set[str] = set()
    source_rows: set[tuple[str, int]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise GateInputError("corpus reference contains a non-object row")
        session = row.get("session")
        request = row.get("request")
        output_ids = row.get("output_ids")
        if not isinstance(session, str) or not session:
            raise GateInputError("corpus row has no source session identity")
        if type(request) is not int or request < 0:
            raise GateInputError(f"corpus row has invalid request index: {session}")
        if (
            not isinstance(output_ids, list)
            or not output_ids
            or any(type(token) is not int for token in output_ids)
        ):
            raise GateInputError(
                f"corpus row has no exact output token vector: {session}/{request}"
            )
        identity = (session, request)
        if identity in source_rows:
            raise GateInputError(f"duplicate corpus identity: {session}/{request}")
        source_rows.add(identity)
        sessions.add(session)

    ordered = sorted(sessions)
    if len(ordered) < size:
        raise GateInputError(
            f"need at least {size} eligible sessions, found {len(ordered)}"
        )
    selected = random.Random(seed).sample(ordered, size)
    selected_rows = sorted(
        (session, request) for session, request in source_rows if session in selected
    )
    return {
        "seed": seed,
        "eligible_session_count": len(ordered),
        "eligible_sessions": ordered,
        "selected_sessions": selected,
        "eligible_source_rows": [
            {"source_file": session, "source_row": request}
            for session, request in selected_rows
        ],
    }


def _nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _occurrence_key(event: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(event[key] for key in OCCURRENCE_FIELDS)


def _event_signature(event: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(event["output_ids"]),
        event["output_length"],
        event["cache_hit_tokens"],
        event["new_prefill_tokens"],
        event["total_retractions"],
    )


def _canonical_json_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_provenance(
    bundle: dict[str, Any], reference: dict[str, Any]
) -> list[str]:
    provenance = bundle.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("schema_version") != 1:
        return ["independently hashed runtime provenance is missing"]
    errors: list[str] = []
    if provenance.get("reference_sha256") != _canonical_json_sha256(reference):
        errors.append(
            "runtime provenance reference SHA-256 differs from the gate reference"
        )

    def valid_artifact(value: Any) -> bool:
        return (
            isinstance(value, dict)
            and isinstance(value.get("path"), str)
            and bool(value["path"])
            and isinstance(value.get("sha256"), str)
            and len(value["sha256"]) == 64
            and all(char in "0123456789abcdef" for char in value["sha256"])
        )

    if not valid_artifact(provenance.get("frozen_plan")):
        errors.append("runtime provenance has no hashed frozen plan")
    receipts = provenance.get("runner_receipts")
    if not isinstance(receipts, dict) or set(receipts) != set(ARMS):
        errors.append(
            "runtime provenance must hash a separate runner receipt for each arm"
        )
    elif any(not valid_artifact(receipts[arm]) for arm in ARMS):
        errors.append("runtime provenance has an invalid runner receipt hash")
    logs = provenance.get("request_logs")
    if not isinstance(logs, dict) or set(logs) != set(ARMS):
        errors.append("runtime provenance must hash request logs for each arm")
    else:
        for arm in ARMS:
            entries = logs[arm]
            if (
                not isinstance(entries, list)
                or not entries
                or any(not valid_artifact(entry) for entry in entries)
            ):
                errors.append(
                    f"runtime provenance has invalid {arm} request log hashes"
                )
    return errors


def _validate_bundle(
    bundle: dict[str, Any],
    reference: dict[str, Any],
    selected_sessions: list[str],
    eligible_source_rows: set[tuple[str, int]],
    reference_outputs: dict[tuple[str, int], tuple[int, ...]],
) -> dict[str, Any]:
    events = bundle.get("events")
    if not isinstance(events, list) or not events:
        return {
            "status": "DEFERRED",
            "errors": [],
            "deferred_reason": "no actual request.finished instrumentation events were supplied",
        }

    arms_present = {
        event.get("arm")
        for event in events
        if isinstance(event, dict) and isinstance(event.get("arm"), str)
    }
    missing_arms = sorted(set(ARMS) - arms_present)
    if missing_arms:
        return {
            "status": "DEFERRED",
            "errors": [],
            "deferred_reason": (
                f"request.finished events are missing arms: {', '.join(missing_arms)}"
            ),
        }

    provenance_errors = _validate_provenance(bundle, reference)
    if provenance_errors:
        return {
            "status": "DEFERRED"
            if provenance_errors
            == ["independently hashed runtime provenance is missing"]
            else "FAIL",
            "errors": []
            if provenance_errors
            == ["independently hashed runtime provenance is missing"]
            else provenance_errors,
            "deferred_reason": provenance_errors[0]
            if len(provenance_errors) == 1
            else None,
        }

    expected_occurrences = bundle.get("expected_occurrences")
    expected_warmup_occurrences = bundle.get("expected_warmup_occurrences")
    drain = bundle.get("drain")
    if not isinstance(expected_occurrences, list) or not expected_occurrences:
        return {
            "status": "DEFERRED",
            "errors": [],
            "deferred_reason": "a frozen expected occurrence ledger was not supplied",
        }
    if not isinstance(expected_warmup_occurrences, list):
        return {
            "status": "DEFERRED",
            "errors": [],
            "deferred_reason": "a frozen expected warmup occurrence ledger was not supplied",
        }
    if not isinstance(drain, dict) or not isinstance(drain.get("by_arm"), dict):
        return {
            "status": "DEFERRED",
            "errors": [],
            "deferred_reason": "per-arm completion receipts were not supplied",
        }

    errors: list[str] = []
    loaded_sessions = bundle.get("loaded_sessions")
    if loaded_sessions != selected_sessions:
        errors.append(
            "loaded_sessions must equal the fixed four-session sample in sample order"
        )

    admissions = bundle.get("profiling_root_admissions")
    if (
        not isinstance(admissions, list)
        or len(admissions) != PROFILE_ROOT_ADMISSION_COUNT
    ):
        errors.append("profiling_root_admissions must contain exactly four roots")
        admissions = []
    admitted_roots: set[tuple[str, str]] = set()
    for admission in admissions:
        if not isinstance(admission, dict):
            errors.append("profiling root admission must be an object")
            continue
        source_file = admission.get("source_file")
        root_id = admission.get("logical_root_id")
        if not isinstance(source_file, str) or source_file not in selected_sessions:
            errors.append(
                "profiling root admission references a session outside the loaded cohort"
            )
            continue
        if not isinstance(root_id, str) or not root_id:
            errors.append("profiling root admission has no logical_root_id")
            continue
        identity = (source_file, root_id)
        if identity in admitted_roots:
            errors.append(
                f"duplicate profiling root admission: {source_file}/{root_id}"
            )
        admitted_roots.add(identity)
    observed_admissions = bundle.get("observed_profiling_root_admissions")
    if not isinstance(observed_admissions, dict) or set(observed_admissions) != set(
        ARMS
    ):
        errors.append(
            "observed profiling root admissions must contain all three AIPerf arms"
        )
    else:
        for arm in ARMS:
            observed = observed_admissions[arm]
            if not isinstance(observed, list) or observed != admissions:
                errors.append(
                    f"{arm} first-seen AIPerf profiling roots differ from frozen admission order"
                )

    by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    arm_fields: dict[str, set[frozenset[str]]] = {arm: set() for arm in ARMS}
    for event in events:
        if not isinstance(event, dict):
            errors.append("event ledger contains a non-object entry")
            continue
        arm = event.get("arm")
        if arm not in ARMS:
            errors.append(f"event has an unsupported arm: {arm!r}")
            continue
        by_arm[arm].append(event)
        arm_fields[arm].add(frozenset(event))
        missing = REQUIRED_EVENT_FIELDS - event.keys()
        if missing:
            errors.append(
                f"{arm} request.finished event is missing fields: {', '.join(sorted(missing))}"
            )
            continue
        if event["event"] != "request.finished":
            errors.append(f"{arm} event name must be request.finished")
        if not isinstance(event["request_id"], str) or not event["request_id"]:
            errors.append(f"{arm} event has no request_id")
        if event["source_file"] not in selected_sessions:
            errors.append(f"{arm} event references a session outside the loaded cohort")
        if (
            isinstance(event["source_file"], str)
            and type(event["source_row"]) is int
            and (event["source_file"], event["source_row"]) not in eligible_source_rows
        ):
            errors.append(
                f"{arm} event references a source row outside the selected corpus identities"
            )
        for key in (
            "phase",
            "logical_root_id",
            "tree_occurrence_id",
            "cache_salt_class",
        ):
            if not isinstance(event[key], str) or not event[key]:
                errors.append(f"{arm} event has invalid {key}")
        for key in (
            "source_row",
            "turn",
            "reset_ordinal",
            "output_length",
            "cache_hit_tokens",
            "new_prefill_tokens",
            "total_retractions",
        ):
            if not _nonnegative_int(event[key]):
                errors.append(f"{arm} event has invalid {key}")
        if (
            not isinstance(event["output_ids"], list)
            or not event["output_ids"]
            or any(type(token) is not int for token in event["output_ids"])
        ):
            errors.append(f"{arm} event has invalid output_ids")
        elif event["output_length"] != len(event["output_ids"]):
            errors.append(f"{arm} output_length does not match output_ids")
        elif (
            isinstance(event["source_file"], str)
            and type(event["source_row"]) is int
            and (event["source_file"], event["source_row"]) in reference_outputs
            and tuple(event["output_ids"])
            != reference_outputs[(event["source_file"], event["source_row"])]
        ):
            errors.append(f"{arm} output_ids differ from the immutable reference")

    schema_sets = [arm_fields[arm] for arm in ARMS]
    if any(len(fields) != 1 for fields in schema_sets):
        errors.append("each arm must emit one consistent request.finished field set")
    elif not (schema_sets[0] == schema_sets[1] == schema_sets[2]):
        errors.append(
            "capture/generated/replay request.finished schemas are asymmetric"
        )
    elif any(
        not REQUIRED_EVENT_FIELDS.issubset(next(iter(fields))) for fields in schema_sets
    ):
        errors.append(
            "request.finished schema omits required occurrence or comparison fields"
        )

    for arm in ARMS:
        profile_roots = set()
        for event in by_arm[arm]:
            source_file = event.get("source_file")
            root_id = event.get("logical_root_id")
            if (
                str(event.get("phase", "")).lower() == "profiling"
                and isinstance(source_file, str)
                and isinstance(root_id, str)
            ):
                profile_roots.add((source_file, root_id))
        if profile_roots != admitted_roots:
            errors.append(
                f"{arm} profiling roots do not match the separate root-admission ledger"
            )

    expected_counts: Counter[tuple[Any, ...]] = Counter()
    for occurrence in expected_occurrences:
        if not isinstance(occurrence, dict):
            errors.append("expected occurrence ledger contains a non-object entry")
            continue
        if not set(OCCURRENCE_FIELDS + ("count",)).issubset(occurrence):
            errors.append(
                "expected occurrence entry is missing identity fields or count"
            )
            continue
        key = tuple(occurrence[field] for field in OCCURRENCE_FIELDS)
        count = occurrence["count"]
        if (
            not isinstance(key[0], str)
            or type(key[1]) is not int
            or any(
                not isinstance(key[index], str) or not key[index]
                for index in (2, 3, 4, 7)
            )
            or any(not _nonnegative_int(key[index]) for index in (5, 6))
            or (key[0], key[1]) not in eligible_source_rows
            or not _nonnegative_int(count)
            or count == 0
        ):
            errors.append("expected occurrence entry has an invalid identity or count")
            continue
        if key in expected_counts:
            errors.append("expected occurrence ledger contains a duplicate identity")
            continue
        expected_counts[key] = count

    expected_warmup_counts: Counter[tuple[Any, ...]] = Counter()
    for occurrence in expected_warmup_occurrences:
        if not isinstance(occurrence, dict):
            errors.append(
                "expected warmup occurrence ledger contains a non-object entry"
            )
            continue
        if not set(OCCURRENCE_FIELDS + ("count",)).issubset(occurrence):
            errors.append(
                "expected warmup occurrence is missing identity fields or count"
            )
            continue
        key = tuple(occurrence[field] for field in OCCURRENCE_FIELDS)
        count = occurrence["count"]
        if (
            not isinstance(key[0], str)
            or type(key[1]) is not int
            or any(
                not isinstance(key[index], str) or not key[index]
                for index in (2, 3, 4, 7)
            )
            or any(not _nonnegative_int(key[index]) for index in (5, 6))
            or (key[0], key[1]) not in eligible_source_rows
            or key[0] not in selected_sessions
            or key[2] != "warmup"
            or not _nonnegative_int(count)
            or count != 1
        ):
            errors.append("expected warmup occurrence has an invalid identity or count")
            continue
        if key in expected_warmup_counts:
            errors.append(
                "expected warmup occurrence ledger contains a duplicate identity"
            )
            continue
        expected_warmup_counts[key] = count

    signatures: dict[str, Counter[tuple[Any, ...]]] = {}
    occurrence_counts: dict[str, Counter[tuple[Any, ...]]] = {}
    for arm in ARMS:
        counts: Counter[tuple[Any, ...]] = Counter()
        occurrences: Counter[tuple[Any, ...]] = Counter()
        request_ids: set[str] = set()
        for event in by_arm[arm]:
            if REQUIRED_EVENT_FIELDS.issubset(event):
                try:
                    counts[(_occurrence_key(event), _event_signature(event))] += 1
                    occurrences[_occurrence_key(event)] += 1
                except TypeError:
                    pass
            request_id = event.get("request_id")
            if isinstance(request_id, str) and request_id in request_ids:
                errors.append(
                    f"{arm} has duplicate request.finished request_id: {request_id}"
                )
            elif isinstance(request_id, str):
                request_ids.add(request_id)
        signatures[arm] = counts
        occurrence_counts[arm] = occurrences
        if occurrences != expected_counts:
            errors.append(
                f"{arm} occurrence identities or multiplicities differ from the frozen ledger"
            )
    if signatures["capture"] != signatures["generated"]:
        errors.append(
            "capture and generated occurrence IDs, outputs, cache metrics, or multiplicities differ"
        )
    if signatures["capture"] != signatures["replay"]:
        errors.append(
            "capture and replay occurrence IDs, outputs, cache metrics, or multiplicities differ"
        )

    raw_warmup_events = bundle.get("warmup_events")
    if not isinstance(raw_warmup_events, list):
        errors.append("warmup_events must be a list")
        raw_warmup_events = []
    warmup_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    warmup_identity_counts: dict[str, Counter[tuple[Any, ...]]] = {
        arm: Counter() for arm in ARMS
    }
    warmup_request_ids: dict[str, set[str]] = {arm: set() for arm in ARMS}
    warmup_signatures: dict[str, Counter[tuple[Any, ...]]] = {}
    raw_expected_warmup_ids = bundle.get("expected_warmup_request_ids")
    if not isinstance(raw_expected_warmup_ids, dict) or set(
        raw_expected_warmup_ids
    ) != set(ARMS):
        errors.append(
            "expected_warmup_request_ids must contain exactly capture, generated, and replay"
        )
        raw_expected_warmup_ids = {}

    warmup_field_sets: dict[str, set[frozenset[str]]] = {arm: set() for arm in ARMS}
    for event in raw_warmup_events:
        if not isinstance(event, dict):
            errors.append("warmup event ledger contains a non-object entry")
            continue
        arm = event.get("arm")
        if arm not in ARMS:
            errors.append(f"warmup event has an unsupported arm: {arm!r}")
            continue
        warmup_by_arm[arm].append(event)
        warmup_field_sets[arm].add(frozenset(event))
        missing = REQUIRED_EVENT_FIELDS - event.keys()
        if missing:
            errors.append(
                f"{arm} warmup event is missing fields: {', '.join(sorted(missing))}"
            )
            continue
        if event["event"] != "request.finished" or event["phase"] != "warmup":
            errors.append(
                f"{arm} warmup event must be request.finished with phase warmup"
            )
        request_id = event["request_id"]
        if not isinstance(request_id, str) or not request_id:
            errors.append(f"{arm} warmup event has no request_id")
        elif request_id in warmup_request_ids[arm]:
            errors.append(
                f"{arm} has duplicate warmup request.finished request_id: {request_id}"
            )
        else:
            warmup_request_ids[arm].add(request_id)
        if (
            not isinstance(event["source_file"], str)
            or event["source_file"] not in selected_sessions
            or type(event["source_row"]) is not int
            or (event["source_file"], event["source_row"]) not in eligible_source_rows
        ):
            errors.append(f"{arm} warmup event references an invalid source row")
        for key in (
            "phase",
            "logical_root_id",
            "tree_occurrence_id",
            "cache_salt_class",
        ):
            if not isinstance(event[key], str) or not event[key]:
                errors.append(f"{arm} warmup event has invalid {key}")
        if event["phase"] == "warmup" and isinstance(event["source_file"], str):
            try:
                identity = _occurrence_key(event)
                warmup_identity_counts[arm][identity] += 1
            except TypeError:
                pass
        for key in (
            "output_length",
            "cache_hit_tokens",
            "new_prefill_tokens",
            "total_retractions",
        ):
            if not _nonnegative_int(event[key]):
                errors.append(f"{arm} warmup event has invalid {key}")
        output_ids = event["output_ids"]
        if (
            not isinstance(output_ids, list)
            or not output_ids
            or any(type(token) is not int for token in output_ids)
        ):
            errors.append(f"{arm} warmup event has invalid output_ids")
        elif event["output_length"] != len(output_ids):
            errors.append(f"{arm} warmup output_length does not match output_ids")
        elif (
            isinstance(event["source_file"], str)
            and type(event["source_row"]) is int
            and (event["source_file"], event["source_row"]) in reference_outputs
            and tuple(output_ids)
            != reference_outputs[(event["source_file"], event["source_row"])]
        ):
            errors.append(
                f"{arm} warmup output_ids differ from the immutable reference"
            )

    for arm in ARMS:
        if warmup_identity_counts[arm] != expected_warmup_counts:
            errors.append(
                f"{arm} warmup identities differ from the frozen warmup ledger"
            )
        expected_ids = raw_expected_warmup_ids.get(arm)
        if (
            not isinstance(expected_ids, list)
            or not all(isinstance(value, str) and value for value in expected_ids)
            or len(expected_ids) != len(set(expected_ids))
        ):
            errors.append(f"{arm} expected warmup request IDs are invalid")
        elif warmup_request_ids[arm] != set(expected_ids):
            errors.append(
                f"{arm} warmup request IDs differ from the frozen warmup ledger"
            )
        arm_warmup_signatures: Counter[tuple[Any, ...]] = Counter()
        for event in warmup_by_arm[arm]:
            if REQUIRED_EVENT_FIELDS.issubset(event):
                try:
                    arm_warmup_signatures[
                        (_occurrence_key(event), _event_signature(event))
                    ] += 1
                except TypeError:
                    pass
        warmup_signatures[arm] = arm_warmup_signatures
    if warmup_signatures["capture"] != warmup_signatures["generated"]:
        errors.append("capture and generated warmup outputs or cache metrics differ")
    if warmup_signatures["capture"] != warmup_signatures["replay"]:
        errors.append("capture and replay warmup outputs or cache metrics differ")
    warmup_schema_sets = [warmup_field_sets[arm] for arm in ARMS]
    if expected_warmup_counts:
        if any(len(fields) != 1 for fields in warmup_schema_sets):
            errors.append("each arm must emit one consistent warmup event field set")
        elif not (
            warmup_schema_sets[0] == warmup_schema_sets[1] == warmup_schema_sets[2]
        ):
            errors.append(
                "capture/generated/replay warmup event schemas are asymmetric"
            )
    elif any(warmup_by_arm[arm] for arm in ARMS):
        errors.append(
            "warmup events are present despite an explicitly empty warmup ledger"
        )

    if not isinstance(drain, dict) or any(key not in drain for key in DRAIN_FIELDS):
        errors.append(
            "drain must report sent, completed, errors, skips, cancellations, and pending"
        )
    else:
        for key in DRAIN_FIELDS:
            if not _nonnegative_int(drain[key]):
                errors.append(f"drain has invalid {key}")
        if _nonnegative_int(drain.get("sent")) and _nonnegative_int(
            drain.get("completed")
        ):
            if drain["sent"] != drain["completed"]:
                errors.append("drain sent and completed counts differ")
        for key in ("errors", "skips", "cancellations", "pending"):
            if _nonnegative_int(drain.get(key)) and drain[key] != 0:
                errors.append(f"drain {key} must be zero")
        arm_drains = drain.get("by_arm")
        if not isinstance(arm_drains, dict) or set(arm_drains) != set(ARMS):
            errors.append(
                "drain by_arm must contain exactly capture, generated, and replay receipts"
            )
        else:
            for arm in ARMS:
                receipt = arm_drains[arm]
                if not isinstance(receipt, dict) or any(
                    key not in receipt for key in DRAIN_FIELDS
                ):
                    errors.append(f"{arm} drain receipt is incomplete")
                    continue
                if any(not _nonnegative_int(receipt[key]) for key in DRAIN_FIELDS):
                    errors.append(f"{arm} drain receipt contains invalid counts")
                    continue
                if receipt["sent"] != receipt["completed"]:
                    errors.append(f"{arm} drain sent and completed counts differ")
                if receipt["completed"] != len(by_arm[arm]) + len(warmup_by_arm[arm]):
                    errors.append(
                        f"{arm} request.finished and warmup counts differ from its completion receipt"
                    )
                for key in ("errors", "skips", "cancellations", "pending"):
                    if receipt[key] != 0:
                        errors.append(f"{arm} drain {key} must be zero")
                phase_drains = receipt.get("phase_drains")
                if not isinstance(phase_drains, dict) or set(phase_drains) != {
                    "warmup",
                    "profiling",
                }:
                    errors.append(
                        f"{arm} drain receipt must include explicit warmup and profiling phase counts"
                    )
                else:
                    for phase, expected_count in (
                        (
                            "warmup",
                            len(raw_expected_warmup_ids.get(arm))
                            if isinstance(raw_expected_warmup_ids.get(arm), list)
                            else -1,
                        ),
                        ("profiling", len(by_arm[arm])),
                    ):
                        phase_receipt = phase_drains[phase]
                        required_phase_fields = (
                            "sent",
                            "completed",
                            "errors",
                            "cancellations",
                            "pending",
                        )
                        if not isinstance(phase_receipt, dict) or any(
                            not _nonnegative_int(phase_receipt.get(field))
                            for field in required_phase_fields
                        ):
                            errors.append(
                                f"{arm} {phase} phase receipt has incomplete counts"
                            )
                            continue
                        if (
                            phase_receipt["sent"] != expected_count
                            or phase_receipt["completed"] != expected_count
                        ):
                            errors.append(
                                f"{arm} {phase} phase counts differ from frozen request IDs"
                            )
                        if any(
                            phase_receipt[field]
                            for field in ("errors", "cancellations", "pending")
                        ):
                            errors.append(
                                f"{arm} {phase} phase errors, cancellations, and pending counts must be zero"
                            )
                    if (
                        sum(
                            phase_drains[phase].get("sent", -1)
                            for phase in ("warmup", "profiling")
                            if isinstance(phase_drains[phase], dict)
                        )
                        != receipt["sent"]
                    ):
                        errors.append(
                            f"{arm} phase sent counts differ from total drain"
                        )
            if all(
                isinstance(arm_drains[arm], dict)
                and all(
                    _nonnegative_int(arm_drains[arm].get(key))
                    for key in ("sent", "completed")
                )
                for arm in ARMS
            ):
                if sum(arm_drains[arm]["sent"] for arm in ARMS) != drain["sent"]:
                    errors.append(
                        "aggregate drain sent count differs from per-arm receipts"
                    )
                if (
                    sum(arm_drains[arm]["completed"] for arm in ARMS)
                    != drain["completed"]
                ):
                    errors.append(
                        "aggregate drain completed count differs from per-arm receipts"
                    )

    return {
        "status": "FAIL" if errors else "PASS",
        "errors": errors,
        "deferred_reason": None,
        "loaded_sessions": loaded_sessions,
        "profiling_root_admissions": admissions,
        "event_counts": {arm: len(by_arm[arm]) for arm in ARMS},
        "occurrence_multiplicity": {arm: sum(signatures[arm].values()) for arm in ARMS},
        "expected_occurrence_multiplicity": sum(expected_counts.values()),
        "drain": drain,
        "provenance": bundle["provenance"],
    }


def evaluate_gate(
    reference: dict[str, Any],
    bundle: dict[str, Any] | None = None,
    *,
    runtime_artifacts_verified: bool = False,
    fixture_only: bool = False,
) -> dict[str, Any]:
    """Validate a bundle; fixture inputs cannot be presented as runtime evidence."""
    cohort = select_cohort(reference)
    if bundle is None:
        validation = {
            "status": "DEFERRED",
            "errors": [],
            "deferred_reason": (
                "actual capture/generated/replay request.finished instrumentation is absent"
            ),
        }
    elif not runtime_artifacts_verified and not fixture_only:
        validation = {
            "status": "DEFERRED",
            "errors": [],
            "deferred_reason": (
                "prebuilt event bundles are fixture-only; runtime evaluation requires "
                "the hashed runtime-manifest path"
            ),
        }
    elif not isinstance(bundle, dict):
        validation = {"status": "FAIL", "errors": ["event bundle must be an object"]}
    elif bundle.get("schema_version") != 2:
        validation = {
            "status": "FAIL",
            "errors": ["event bundle schema_version must be 2"],
        }
    else:
        eligible_source_rows = {
            (row["source_file"], row["source_row"])
            for row in cohort["eligible_source_rows"]
        }
        reference_outputs = {
            (row["session"], row["request"]): tuple(row["output_ids"])
            for row in reference["rows"]
            if row["session"] in cohort["selected_sessions"]
        }
        validation = _validate_bundle(
            bundle,
            reference,
            cohort["selected_sessions"],
            eligible_source_rows,
            reference_outputs,
        )
    return {
        "schema_version": 1,
        "gate": "B2 CPU preflight",
        "runtime_artifacts_verified": runtime_artifacts_verified,
        "evidence_scope": (
            "runtime"
            if runtime_artifacts_verified
            else "fixture-only"
            if fixture_only
            else "unverified"
        ),
        **cohort,
        **validation,
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise GateInputError(f"expected a JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        reference = _read_json(args.reference)
        if args.runtime_manifest:
            manifest = _read_json(args.runtime_manifest)
            bundle = build_event_bundle(
                manifest, args.runtime_manifest.resolve().parent, reference
            )
        else:
            bundle = None
        result = evaluate_gate(
            reference,
            bundle,
            runtime_artifacts_verified=args.runtime_manifest is not None,
        )
    except (OSError, json.JSONDecodeError, GateInputError, RuntimeLedgerError) as error:
        result = {
            "schema_version": 1,
            "gate": "B2 CPU preflight",
            "status": "FAIL",
            "errors": [str(error)],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "gate",
                    "status",
                    "selected_sessions",
                    "errors",
                    "deferred_reason",
                )
            }
        )
    )
    return 0 if result["status"] in {"PASS", "DEFERRED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
