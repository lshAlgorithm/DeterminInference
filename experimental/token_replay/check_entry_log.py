from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def check_entry_log(log: Path, traces: Path, mode: str) -> dict:
    """Compare actual OpenAI inputs and engine output IDs with qualified source rows."""
    expected = {}
    for path in sorted(traces.glob("*.json")):
        trace = json.loads(path.read_text())
        rows = []
        for row in trace["requests"]:
            rows.extend(row["requests"] if row["type"] == "subagent" else [row])
        for row in rows:
            key = tuple(row["input_ids"])
            value = row["output_ids"]
            if key in expected and expected[key] != value:
                raise ValueError(
                    "ambiguous fixture: identical prompt with different outputs"
                )
            expected[key] = value
    counts = Counter()
    salts = set()
    failures = []
    completed = []
    for line in log.open():
        if not line.startswith("{"):
            continue
        event = json.loads(line)
        obj = event.get("obj", {})
        if event.get("event") == "request.received.openai":
            counts["wire_requests"] += 1
            source = expected.get(tuple(obj.get("prompt", [])))
            cap = obj.get("max_tokens")
            if source is None or cap not in (1, len(source)):
                failures.append(
                    {"kind": "wire_ids_or_length", "request": counts["wire_requests"]}
                )
                continue
            replay = obj.get("replay_token_ids")
            if (mode == "recorded" and replay != source[:cap]) or (
                mode == "generated" and replay is not None
            ):
                failures.append(
                    {"kind": "wire_replay_ids", "request": counts["wire_requests"]}
                )
            salt = obj.get("cache_salt")
            if not salt:
                failures.append(
                    {"kind": "missing_cache_salt", "request": counts["wire_requests"]}
                )
            else:
                salts.add(salt)
            counts["one_token_requests" if cap == 1 else "full_length_requests"] += 1
        if event.get("event") != "request.finished":
            continue
        source = expected.get(tuple(obj.get("input_ids") or []))
        if source is None:
            continue
        counts["engine_completions"] += 1
        out = event["out"]
        meta = out["meta_info"]
        cap = obj["sampling_params"]["max_new_tokens"]
        if meta["finish_reason"]["type"] != "length":
            failures.append({"kind": "incomplete_request", "rid": event["rid"]})
        elif mode == "recorded" and out["output_ids"] != source[:cap]:
            failures.append({"kind": "engine_output_ids", "rid": event["rid"]})
        counts["prompt_tokens"] += meta["prompt_tokens"]
        counts["cached_tokens"] += meta["cached_tokens"]
        counts["output_tokens"] += meta["completion_tokens"]
        completed.append(
            {
                "rid": event["rid"],
                "prompt_tokens": meta["prompt_tokens"],
                "cached_tokens": meta["cached_tokens"],
                "output_tokens": meta["completion_tokens"],
                "salt": obj.get("extra_key"),
                "finish_reason": meta["finish_reason"],
            }
        )
    if (
        counts["wire_requests"] == 0
        or counts["wire_requests"] != counts["engine_completions"]
    ):
        failures.append(
            {
                "kind": "wire_completion_count",
                "wire": counts["wire_requests"],
                "completed": counts["engine_completions"],
            }
        )
    return {
        "mode": mode,
        "counts": dict(counts),
        "salt_count": len(salts),
        "failures": failures,
        "completed": completed,
        "passed": not failures,
        "scope": "wire and output ID audit; DAG event ordering requires harness records",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--mode", choices=("recorded", "generated"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check_entry_log(args.log, args.traces, args.mode)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "completed"},
            indent=2,
        )
    )
    raise SystemExit(0 if result["passed"] else 1)
