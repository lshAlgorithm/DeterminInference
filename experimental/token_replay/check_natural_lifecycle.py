from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from check_retraction_lifecycle import read_observer_events, validate_lifecycle


def check_natural_lifecycle(receipt_path: Path, observer_prefix: Path) -> dict:
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("natural_trigger") != "kv_full":
        raise ValueError("receipt must declare the natural kv_full trigger")
    events = read_observer_events(observer_prefix)
    result = validate_lifecycle(receipt, events)
    long = next((row for row in receipt.get("requests", []) if row.get("role") == "long"), None)
    chain = result["L1"].get("evidence", [])
    rid = long["rid"] if long else None
    natural_events = [
        event
        for event in events
        if event.get("event") == "retraction_trigger"
        and event.get("trigger") == "kv_full"
        and any(row.get("rid") == rid for row in event.get("requests", []))
    ]
    if result["L1"]["status"] == "PASS":
        reset_pid, reset_seq = chain[0]
        if not any(
            event["pid"] == reset_pid and event["seq"] < reset_seq
            for event in natural_events
        ):
            result["L1"] = {
                "status": "UNCOVERED",
                "evidence": [],
                "errors": ["L1 chain has no preceding same-request kv_full trigger"],
            }
    result["natural_trigger_events"] = [
        [event["pid"], event["seq"]] for event in natural_events
    ]
    result["validator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--observer-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check_natural_lifecycle(args.receipt, args.observer_prefix)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["L1"]["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
