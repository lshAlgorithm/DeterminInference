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


def validate_lifecycle(receipt: dict, events: list[dict]) -> dict:
    result = {case: {"status": "UNCOVERED", "evidence": [], "errors": []} for case in ("L1", "L2", "L3")}
    records = receipt.get("requests", [])
    by_rid = {r["rid"]: r for r in records}
    if len(by_rid) != len(records):
        raise ValueError("duplicate request IDs in receipt")
    if any(a.startswith(b) for a in by_rid for b in by_rid if a != b):
        raise ValueError("request IDs must not prefix-match each other")
    checks = {}
    for record in records:
        case = "L3" if record["role"].startswith(("cancel", "after_cancel", "fresh")) else "L2"
        try:
            if record.get("error"):
                raise ValueError(record["error"])
            count, terminal = 0, None
            for wire in record.get("packets", []):
                if terminal is not None:
                    raise ValueError("packet after terminal response")
                state = validate_packet(wire["packet"], record["rid"], record["expected"], count)
                count = state["count"]
                if state["terminal"]:
                    terminal = state
            if terminal is None:
                result[case]["errors"].append(f"no terminal response: {record['rid']}")
                continue
            checks[record["rid"]] = validate_request_events(record, events)
            if record["role"] != "cancel" and terminal["abort"]:
                raise ValueError("unexpected abort of a noncancelled request")
            if record["role"] == "cancel" and terminal["abort"]:
                backend = checks[record["rid"]]["terminal"]
                aborted = [(e, s) for e, s, reason in backend if reason.get("type") == "abort"]
                if aborted:
                    if any(snap["output_len"] != count for _, snap in aborted):
                        raise ValueError("abort receipt disagrees with backend retained count")
                record["validated_abort_count"] = count
        except (ValueError, KeyError, TypeError) as error:
            result[case]["status"] = "FAIL"
            result[case]["errors"].append(f"{record['rid']}: {error}")
            if record["role"] == "long":
                result["L1"]["status"] = "FAIL"
                result["L1"]["errors"].append(str(error))
    long = next((r for r in records if r["role"] == "long"), None)
    if long and checks.get(long["rid"], {}).get("chain") and result["L1"]["status"] != "FAIL":
        result["L1"] = {"status": "PASS", "evidence": checks[long["rid"]]["chain"], "errors": []}
    filters, merges = [], []
    for event in events:
        if event["event"] not in ("filter_batch", "merge_batch"):
            continue
        before = {s["rid"]: s for s in event.get("before", [])}
        after = {s["rid"]: s for s in event.get("after", [])}
        other = {s["rid"]: s for s in event.get("other", [])}
        involved = (set(before) | set(after) | set(other)) & set(by_rid)
        if not involved:
            continue
        for rid in involved:
            record = by_rid[rid]
            for snap in (before.get(rid), after.get(rid), other.get(rid)):
                if snap is not None and record["replay"] and (snap.get("replay_slot") is None or snap["replay_slot"]["length"] != len(record["expected"])):
                    result["L2"]["status"] = "FAIL"
                    result["L2"]["errors"].append(f"wrong active replay metadata mapping: {rid}")
        if event["event"] == "filter_batch" and set(before) != set(after) and any(by_rid[r]["replay"] for r in involved):
            filters.append([event["pid"], event["seq"]])
        if event["event"] == "merge_batch" and other and set(after) == set(before) | set(other) and set(after) != set(before) and any(by_rid[r]["replay"] for r in involved):
            merges.append([event["pid"], event["seq"]])
    mix = [r for r in records if r["role"] in ("long", "short", "ordinary", "after_mix")]
    if filters and merges and any(r["role"] == "ordinary" for r in mix) and any(r["role"] == "after_mix" for r in mix) and all(r["rid"] in checks for r in mix) and result["L2"]["status"] != "FAIL":
        result["L2"]["status"] = "PASS"
        result["L2"]["evidence"] = filters + merges
    cancel = next((r for r in records if r["role"] == "cancel"), None)
    followups = [r for r in records if r["role"] in ("after_cancel", "fresh")]
    if cancel and "validated_abort_count" in cancel and result["L3"]["status"] != "FAIL":
        aborts = [(e, s) for e, s, reason in checks[cancel["rid"]]["terminal"] if reason.get("type") == "abort"]
        sent = cancel.get("abort_receipt", {})
        active = 0 < sent.get("observed_count", 0) <= cancel["validated_abort_count"] < len(cancel["expected"])
        retired = []
        if aborts:
            terminal, _ = aborts[0]
            retired = [e for e in events if e["pid"] == terminal["pid"] and e["seq"] >= terminal["seq"] and
                       ((e["event"] == "abort_request" and e.get("phase") == "waiting_removed" and any(s["rid"] == cancel["rid"] for s in e.get("requests", []))) or
                        (e["event"] == "filter_batch" and any(s["rid"] == cancel["rid"] for s in e.get("before", [])) and all(s["rid"] != cancel["rid"] for s in e.get("after", []))))]
            if retired:
                retirement = retired[0]
                for later in events:
                    if later["pid"] != retirement["pid"] or later["seq"] <= retirement["seq"]:
                        continue
                    active_members = list(later.get("after", []))
                    if later["event"] in {"run_batch_dispatch", "reset_for_retract", "retraction_trigger"}:
                        active_members += later.get("requests", [])
                    if any(s["rid"] == cancel["rid"] for s in active_members):
                        result["L3"]["status"] = "FAIL"
                        result["L3"]["errors"].append("cancelled request reappeared after retirement")
                        break
                    if later["event"] == "process_batch_result":
                        for snap in later.get("requests", []):
                            if snap["rid"] == cancel["rid"] and (snap.get("finished_reason") or {}).get("type") != "abort":
                                result["L3"]["status"] = "FAIL"
                                result["L3"]["errors"].append("post-retirement completion is not an abort")
                                break
        if active and sent.get("status") == 200 and aborts and retired and len(followups) == 2 and all(r["rid"] in checks for r in followups) and result["L3"]["status"] != "FAIL":
            result["L3"]["status"] = "PASS"
            result["L3"]["evidence"] = [[e["pid"], e["seq"]] for e in [aborts[0][0], retired[0]]]
    for case in result.values():
        if case["status"] == "UNCOVERED" and not case["errors"]:
            case["errors"].append("required same-request lifecycle or completion witness absent")
    return result


def make_request(spec: dict, concurrency: int, role: str, source_index: int,
                 replay: bool, ordinary: list[int] | None = None) -> dict:
    row = spec["selected"][source_index]
    request = row["request"]
    tag = canonical_hash([spec["spec_sha256"], concurrency, role, source_index])[:24]
    rid = "b1-" + tag
    params = {**spec["sampling"], "max_new_tokens": row["expected_length"]}
    if replay:
        params["replay_token_ids"] = request["output_ids"]
    return {"rid": rid, "role": role, "source_index": source_index, "replay": replay,
            "input_ids": request["input_ids"],
            "expected": request["output_ids"] if replay else ordinary,
            "payload": {"rid": rid, "input_ids": request["input_ids"], "extra_key": "salt-" + tag,
                        "sampling_params": params, "stream": True,
                        "incremental_streaming_output": False},
            "packets": [], "terminal": False, "first_partial_count": 0}


def stream_request(url: str, record: dict, messages: queue.Queue, deadline_ns: int,
                   raw_path: Path) -> None:
    rid = record["rid"]
    request = urllib.request.Request(url.rstrip("/") + "/generate",
                                     json.dumps(record["payload"]).encode(),
                                     {"Content-Type": "application/json"})
    try:
        remaining = (deadline_ns - time.time_ns()) / 1e9
        if remaining <= 0:
            raise TimeoutError("control window expired before dispatch")
        with raw_path.open("x") as raw, urllib.request.urlopen(request, timeout=min(10, remaining)) as response:
            total = 0
            for line in response:
                timestamp = time.time_ns()
                total += len(line)
                if len(line) > 2 * 1024 * 1024 or total > 128 * 1024 * 1024:
                    raise ValueError("stream exceeded bounded receipt size")
                raw.write(json.dumps({"time_ns": timestamp, "raw": line.decode("utf-8")}) + "\n")
                raw.flush()
                if timestamp >= deadline_ns:
                    raise TimeoutError("control window expired while streaming")
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                packet = json.loads(data)
                messages.put({"rid": rid, "packet": packet, "time_ns": timestamp})
        messages.put({"rid": rid, "closed": True, "time_ns": time.time_ns()})
    except Exception as error:
        messages.put({"rid": rid, "error": f"{type(error).__name__}: {error}", "time_ns": time.time_ns()})


def receive_message(messages: queue.Queue, records: dict[str, dict], deadline_ns: int) -> None:
    remaining = (deadline_ns - time.time_ns()) / 1e9
    if remaining <= 0:
        raise TimeoutError("bounded lifecycle control window expired")
    try:
        message = messages.get(timeout=min(remaining, 0.25))
    except queue.Empty:
        return
    record = records[message["rid"]]
    if "error" in message:
        record["error"] = message["error"]
        record["closed"] = True
        raise RuntimeError(message["error"])
    if message.get("closed"):
        record["closed"] = True
        if not record["terminal"]:
            raise RuntimeError("generate stream closed without a terminal receipt")
        return
    record["packets"].append({"time_ns": message["time_ns"], "packet": message["packet"]})
    if record["terminal"]:
        raise ValueError("stream emitted a packet after its terminal receipt")
    state = validate_packet(message["packet"], record["rid"], record["expected"], record.get("count", 0))
    record["count"], record["terminal"] = state["count"], state["terminal"]
    if state["count"] and not state["terminal"] and not record["first_partial_count"]:
        record["first_partial_count"] = state["count"]
        record["first_partial_time_ns"] = message["time_ns"]
    if state["terminal"]:
        record["terminal_time_ns"] = message["time_ns"]
        record["aborted"] = state["abort"]


def abort_owned_request(url: str, record: dict, deadline_ns: int, cleanup: bool = False) -> dict:
    if not record["rid"].startswith("b1-"):
        raise ValueError("refusing cancellation outside this driver's request namespace")
    remaining = (deadline_ns - time.time_ns()) / 1e9
    if remaining <= 0:
        return {"error": "no allocation time remains for cancellation", "cleanup": cleanup}
    receipt = {"rid": record["rid"], "sent_time_ns": time.time_ns(),
               "observed_count": record.get("count", 0), "cleanup": cleanup}
    request = urllib.request.Request(url.rstrip("/") + "/abort_request",
                                     json.dumps({"rid": record["rid"], "abort_all": False}).encode(),
                                     {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=min(10, remaining)) as response:
            receipt.update(status=response.status, body=response.read(1024 * 1024).decode())
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
    receipt["response_time_ns"] = time.time_ns()
    return receipt


