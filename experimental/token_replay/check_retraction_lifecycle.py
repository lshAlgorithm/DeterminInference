from __future__ import annotations

import argparse
import hashlib
import json
import queue
import random
import tarfile
import threading
import time
import urllib.request
from pathlib import Path


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_qualified_rows(archive: Path, manifest_path: Path, reference_path: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text())
    reference = json.loads(reference_path.read_text())
    if manifest.get("status") != "qualified" or manifest.get("selected_requests") != 447:
        raise ValueError("expected the qualified 447-request Dense source manifest")
    if not reference.get("complete") or reference.get("requests_exact") != 447:
        raise ValueError("the frozen corpus reference must cover all 447 requests")
    references = {(r["session"], r["request"]): r for r in reference["rows"]}
    if len(references) != 447:
        raise ValueError("duplicate or missing corpus reference identities")
    rows = []
    with tarfile.open(archive) as source:
        for selected in manifest["selected"]:
            member = "exgentic-dense-02/" + selected["file"]
            stream = source.extractfile(member)
            if stream is None:
                raise ValueError(f"missing source member: {member}")
            raw = stream.read(16 * 1024 * 1024 + 1)
            if len(raw) > 16 * 1024 * 1024:
                raise ValueError("source session exceeds bounded input size")
            if hashlib.sha256(raw).hexdigest() != selected["sha256"]:
                raise ValueError(f"source manifest hash mismatch: {member}")
            requests = json.loads(raw)["requests"]
            if len(requests) != len(selected["requests"]):
                raise ValueError("source request count differs from manifest")
            for index, request in enumerate(requests):
                identity = (Path(selected["file"]).name, index)
                expected = references.get(identity)
                if expected is None or request["output_ids"] != expected["output_ids"]:
                    raise ValueError(f"source/reference output mismatch: {identity}")
                if request.get("type") != "s" or request["in"] != len(request["input_ids"]) or request["out"] != len(request["output_ids"]):
                    raise ValueError(f"incomplete source request: {identity}")
                vectors = request["input_ids"] + request["output_ids"]
                if any(type(token) is not int or token < 0 for token in vectors):
                    raise ValueError("invalid source token IDs")
                rows.append({"source_file": member, "source_row": index,
                             "source_sha256": selected["sha256"],
                             "source_session_id": selected["session_id"],
                             "source_span": selected["requests"][index],
                             "request": request})
    if len(rows) != 447 or len({(r["source_file"], r["source_row"]) for r in rows}) != 447:
        raise ValueError("source denominator must be 447 distinct complete requests")
    return rows


def build_spec(rows: list[dict], identity: dict) -> dict:
    required = ("model_sha256", "serving_source_sha256", "observer_sha256", "wheel_sha256")
    for key in required:
        value = identity.get(key, "")
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"identity requires a SHA256: {key}")
    eligible = sorted((r for r in rows if len(r["request"]["output_ids"]) >= 64
                       and len(r["request"]["input_ids"]) + len(r["request"]["output_ids"]) <= 16384),
                      key=lambda r: (r["source_file"], r["source_row"]))
    random.Random(42).shuffle(eligible)
    selected, lengths = [], set()
    for row in eligible:
        length = len(row["request"]["output_ids"])
        if length in lengths:
            continue
        lengths.add(length)
        selected.append({**row, "expected_length": length,
                         "input_sha256": canonical_hash(row["request"]["input_ids"]),
                         "output_sha256": canonical_hash(row["request"]["output_ids"])})
        if len(selected) == 8:
            break
    if len(selected) != 8:
        raise ValueError("eight distinct eligible lengths are required before execution")
    spec = {"schema_version": 1, "seed": 42, "identity": identity,
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_requests": len(rows), "selected": selected,
            "model": "Qwen2.5-7B-Instruct-GPTQ-Int4",
            "server": {"context_length": 16384, "max_total_tokens": 131072,
                       "max_running_requests": 16, "cuda_graph_max_bs": 16,
                       "attention_backend": "triton", "random_seed": 42,
                       "disable_cuda_graph": True},
            "trigger": {"SGLANG_TEST_RETRACT": "true", "SGLANG_TEST_RETRACT_INTERVAL": "3",
                        "SGLANG_TEST_RETRACT_NO_PREFILL_BS": "2147483648"},
            "attempts": [4, 8], "attempt_seconds": 600, "allocation_seconds": 1800,
            "io_timeout_seconds": 10, "cleanup_seconds": 30,
            "sampling": {"temperature": 0, "ignore_eos": True, "sampling_seed": 42},
            "ordinary_source_index": min(range(4), key=lambda i: selected[i]["expected_length"]),
            "schedule": {"mix": "longest replay, then c-2 shorter replays and one ordinary; peak c",
                         "cancel": "remaining selected source row, after mixed requests drain",
                         "followup": "ordinary and same-source fresh replay after terminal abort and retirement"},
            "scope": "controlled lifecycle correctness only; no timing or KV-equality claim"}
    spec["spec_sha256"] = canonical_hash(spec)
    return spec


def validate_packet(packet: dict, rid: str, expected: list[int] | None, last_count: int) -> dict:
    meta = packet.get("meta_info", {})
    if meta.get("id") != rid:
        raise ValueError("stream request-ID mismatch")
    ids, count = packet.get("output_ids"), meta.get("completion_tokens")
    if not isinstance(ids, list) or any(type(x) is not int or x < 0 for x in ids):
        raise ValueError("stream output_ids must be nonnegative integers")
    if type(count) is not int or count < last_count:
        raise ValueError("stream completion count moved backwards")
    reason = meta.get("finish_reason")
    abort = isinstance(reason, dict) and reason.get("type") == "abort"
    if expected is not None and count > len(expected):
        raise ValueError("stream exceeded the fixed output budget")
    if abort:
        if expected is not None:
            allowed = [expected[:count]]
            if count:
                allowed.append(expected[count - 1:count])
            if ids not in allowed:
                raise ValueError("abort IDs disagree with its retained prefix count")
    else:
        if len(ids) != count:
            raise ValueError("normal streaming packets must contain cumulative output IDs")
        if expected is not None and ids != expected[:count]:
            raise ValueError("stream output prefix mismatch")
        if reason is not None and expected is not None and (ids != expected or reason.get("type") != "length"):
            raise ValueError("successful stream did not finish with the complete expected vector")
    return {"count": count, "terminal": reason is not None, "abort": abort}


def read_observer_events(prefix: Path) -> list[dict]:
    events, seen, last = [], set(), {}
    for path in sorted(prefix.parent.glob(prefix.name + ".*.jsonl")):
        if path.stat().st_size > 256 * 1024 * 1024:
            raise ValueError("observer file exceeds bounded input size")
        with path.open() as source:
            for line in source:
                event = json.loads(line)
                key = (event.get("pid"), event.get("seq"))
                if event.get("schema_version") != 1 or type(key[0]) is not int or type(key[1]) is not int:
                    raise ValueError("invalid lifecycle observer event identity")
                if key in seen or key[1] <= last.get(key[0], -1):
                    raise ValueError("duplicate or nonmonotonic lifecycle observer event")
                seen.add(key)
                last[key[0]] = key[1]
                events.append(event)
    return sorted(events, key=lambda e: (e["time_ns"], e["pid"], e["seq"]))


def validate_request_events(record: dict, events: list[dict]) -> dict:
    rid, expected, prompt = record["rid"], record["expected"], record["input_ids"]
    relevant, terminal, resets, resumes, decodes, progress = [], [], [], [], [], []
    for event in events:
        for snap in event.get("requests", []):
            if snap["rid"] != rid:
                continue
            relevant.append((event, snap))
            ids, count = snap["output_ids"], snap["output_len"]
            if count != len(ids) or ids != expected[:count] or count > len(expected):
                raise ValueError(f"backend prefix mismatch: {rid}")
            if "origin_input_ids" in snap and snap["origin_input_ids"] != prompt:
                raise ValueError(f"backend original prompt mismatch: {rid}")
            if snap.get("output_sha256") not in (None, canonical_hash(ids)):
                raise ValueError("backend prefix hash mismatch")
            reason = snap.get("finished_reason") or event.get("terminal_reason")
            if reason is not None:
                terminal.append((event, snap, reason))
            if event["event"] == "reset_for_retract" and count:
                resets.append((event, snap))
            if event["event"] == "run_batch_dispatch":
                mode = event["forward_mode"].lower()
                if "extend" in mode and snap["retraction_count"] > 0:
                    fill = prompt + ids
                    cached = snap["cached_prefix_length"]
                    if snap.get("origin_input_ids") != prompt or snap["fill_ids"] != fill or type(cached) is not int or not 0 <= cached <= len(fill):
                        raise ValueError("resumed input must be original prompt plus retained output prefix")
                    if snap["extend_ids"] != fill[cached:] or snap["extend_input_len"] != len(fill) - cached:
                        raise ValueError("resumed cached prefix or actual extend suffix mismatch")
                    resumes.append((event, snap))
                if "decode" in mode:
                    decodes.append((event, snap))
            if event["event"] == "process_batch_result":
                progress.append((event, snap))
    chain = None
    for reset, old in resets:
        for resume, retained in resumes:
            if resume["pid"] != reset["pid"] or resume["seq"] <= reset["seq"] or retained["output_ids"] != old["output_ids"]:
                continue
            continued = [(e, s) for e, s in progress if e["pid"] == reset["pid"] and e["seq"] > resume["seq"] and s["output_len"] > old["output_len"]]
            later_decode = [(e, s) for e, s in decodes if e["pid"] == reset["pid"] and e["seq"] > resume["seq"] and s["output_len"] > old["output_len"]]
            finished = [(e, s, r) for e, s, r in terminal if e["pid"] == reset["pid"] and e["seq"] > resume["seq"] and s["output_ids"] == expected and r.get("type") == "length"]
            if continued and later_decode and finished and record.get("first_partial_count", 0) <= old["output_len"] and record.get("first_partial_count", 0) > 0:
                chain = [[e["pid"], e["seq"]] for e in (reset, resume, continued[0][0], later_decode[0][0], finished[0][0])]
                break
        if chain:
            break
    return {"relevant": relevant, "terminal": terminal, "chain": chain}


