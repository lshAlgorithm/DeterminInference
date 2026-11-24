from __future__ import annotations

import argparse
import json
import queue
import time
import urllib.request
from pathlib import Path

from check_retraction_lifecycle import (
    abort_owned_request,
    canonical_hash,
    launch_request,
    make_request,
    read_observer_events,
    receive_message,
    validate_request_events,
)


def control(url: str, endpoint: str, payload: dict[str, str]) -> dict:
    request = urllib.request.Request(
        url.rstrip("/") + endpoint,
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        result = {"status": response.status, "body": json.load(response)}
    if result["status"] != 200 or result["body"].get("status") != "ok":
        raise ValueError(f"control request failed: {result}")
    return result


def run_pause_l1(spec_path: Path, url: str, observer_prefix: Path, output: Path) -> dict:
    spec = json.loads(spec_path.read_text())
    digest = spec.pop("spec_sha256")
    if canonical_hash(spec) != digest or spec["seed"] != 42:
        raise ValueError("frozen source specification mismatch")
    spec["spec_sha256"] = digest
    source_index = max(range(len(spec["selected"])), key=lambda i: spec["selected"][i]["expected_length"])
    record = make_request(spec, 1, "pause_l1", source_index, True)
    record["rid"] = "b1-" + canonical_hash([digest, "pause_l1", source_index, output.name])[:24]
    record["payload"]["rid"] = record["rid"]
    record["payload"]["extra_key"] = "salt-" + record["rid"]
    output.mkdir(parents=True, exist_ok=False)
    deadline = time.time_ns() + 180_000_000_000
    messages: queue.Queue = queue.Queue()
    records: dict[str, dict] = {}
    paused = False
    receipt = {
        "seed": 42,
        "source_spec_sha256": digest,
        "source_index": source_index,
        "request_id": record["rid"],
        "expected_length": len(record["expected"]),
        "scope": "controlled pause/retract correctness; no performance claim",
        "status": "INCOMPLETE",
    }
    worker = launch_request(url, record, records, messages, deadline, output)
    try:
        while record.get("count", 0) < 8 and not record["terminal"]:
            receive_message(messages, records, deadline)
        if record["terminal"]:
            raise ValueError("request completed before the pause control point")
        receipt["pause_after_observed_tokens"] = record["count"]
        receipt["pause"] = control(url, "/pause_generation", {"mode": "retract"})
        paused = True
        until = min(deadline, time.time_ns() + 10_000_000_000)
        while time.time_ns() < until:
            events = read_observer_events(observer_prefix)
            if any(e["event"] == "reset_for_retract" and any(s["rid"] == record["rid"] for s in e["requests"]) for e in events):
                break
            time.sleep(0.05)
        else:
            raise ValueError("selected request had no reset_for_retract event")
        receipt["continue"] = control(url, "/continue_generation", {})
        paused = False
        while not record["terminal"] or not record.get("closed"):
            receive_message(messages, records, deadline)
        if record.get("aborted") or record["count"] != len(record["expected"]):
            raise ValueError("replayed request did not finish at its recorded length")
        events = read_observer_events(observer_prefix)
        check = validate_request_events(record, events)
        if not check["chain"]:
            raise ValueError("same-request reset, resume, decode and finish chain absent")
        receipt["chain"] = check["chain"]
        receipt["status"] = "PASS"
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        receipt["status"] = "FAIL"
    finally:
        if paused:
            try:
                receipt["cleanup_continue"] = control(url, "/continue_generation", {})
            except Exception as error:
                receipt["cleanup_continue_error"] = str(error)
        if not record["terminal"]:
            receipt["cleanup_abort"] = abort_owned_request(url, record, time.time_ns() + 10_000_000_000, cleanup=True)
        worker.join(timeout=10)
        receipt["observed_tokens"] = record.get("count", 0)
        receipt["closed"] = record.get("closed", False)
        with (output / "receipt.json").open("x") as stream:
            json.dump(receipt, stream, indent=2)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--observer-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_pause_l1(args.spec, args.url, args.observer_prefix, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "chain"}, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
