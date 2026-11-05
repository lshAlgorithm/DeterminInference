from __future__ import annotations

import argparse
import json
import math
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from infx.bench.env import InputError


def load_sessions(path: Path) -> dict[str, list[dict]]:
    """Load ordered, full-token turns grouped by independent session."""
    sessions: dict[str, list[dict]] = {}
    with path.open() as source:
        for number, line in enumerate(source, 1):
            row = json.loads(line)
            required = {"session_id", "input_ids", "output_ids", "delay_seconds"}
            if not isinstance(row, dict) or set(row) != required:
                raise InputError(f"line {number}: expected exactly {sorted(required)}")
            if not isinstance(row["session_id"], str) or not row["session_id"]:
                raise InputError(f"line {number}: session_id must be a nonempty string")
            for field in ("input_ids", "output_ids"):
                ids = row[field]
                if (
                    not isinstance(ids, list)
                    or not ids
                    or any(type(t) is not int or t < 0 for t in ids)
                ):
                    raise InputError(f"line {number}: {field} must contain nonnegative integer IDs")
            delay = row["delay_seconds"]
            if type(delay) not in (int, float) or not math.isfinite(delay) or delay < 0:
                raise InputError(f"line {number}: delay_seconds must be finite and nonnegative")
            sessions.setdefault(row["session_id"], []).append(row)
    if not sessions:
        raise InputError("the trace contains no sessions")
    return sessions


def replay_session(url: str, turns: list[dict], timeout: float, seed: int) -> list[dict]:
    """Send each turn after its predecessor completes and verify every output ID."""
    address = urllib.parse.urlsplit(url)
    if address.scheme not in {"http", "https"} or not address.hostname:
        raise InputError("the serving URL must use HTTP or HTTPS")
    results = []
    for index, turn in enumerate(turns):
        time.sleep(turn["delay_seconds"])
        payload = {
            "input_ids": turn["input_ids"],
            "sampling_params": {
                "temperature": 0,
                "ignore_eos": True,
                "sampling_seed": seed,
                "max_new_tokens": len(turn["output_ids"]),
                "replay_token_ids": turn["output_ids"],
            },
        }
        request = urllib.request.Request(  # noqa: S310 — HTTP(S) validated above.
            url.rstrip("/") + "/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            output = json.load(response)
        elapsed = time.perf_counter() - start
        if output.get("output_ids") != turn["output_ids"]:
            raise InputError(
                f"session {turn['session_id']} turn {index}: output IDs do not match trace"
            )
        results.append(
            {
                "session_id": turn["session_id"],
                "turn": index,
                "input_tokens": len(turn["input_ids"]),
                "output_tokens": len(turn["output_ids"]),
                "request_seconds": elapsed,
                "meta_info": output.get("meta_info", {}),
            }
        )
    return results


def main(argv: list[str]) -> int:
    """Replay independent sessions with bounded concurrency and explicit inputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--url")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--agentx-model")
    args = parser.parse_args(argv)
    if args.seed < 0:
        raise InputError("seed must be nonnegative")
    sessions = load_sessions(args.trace)
    if args.agentx_model is not None:
        export_agentx(sessions, args.output, args.agentx_model, args.seed)
        return 0
    if args.url is None or args.concurrency is None or args.timeout is None:
        raise InputError("native replay requires url, concurrency, and timeout")
    if args.concurrency <= 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        raise InputError("concurrency and timeout must be positive")
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(replay_session, args.url, turns, args.timeout, args.seed)
            for turns in sessions.values()
        ]
        results = [result for future in futures for result in future.result()]
    args.output.write_text(
        json.dumps(
            {
                "unofficial": True,
                "seed": args.seed,
                "concurrency": args.concurrency,
                "trace": str(args.trace),
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    return 0
