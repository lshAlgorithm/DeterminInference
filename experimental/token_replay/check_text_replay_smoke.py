from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.request
from pathlib import Path
from typing import TextIO


PROMPT = (
    "Explain why a clear sky appears blue in two short sentences. "
    "Use ordinary words and avoid lists. "
) * 12
SUFFIX = "\nNow summarize the answer in one sentence."
SEED = 42


def post(url: str, path: str, payload: dict | None = None) -> dict:
    request = urllib.request.Request(
        url.rstrip("/") + path,
        json.dumps(payload).encode() if payload is not None else b"{}",
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        if path == "/flush_cache":
            body = response.read().decode()
            assert response.status == 200 and body.startswith("Cache flushed."), body
            return {}
        return json.load(response)


def generate(
    url: str,
    input_ids: list[int],
    output_length: int,
    replay: list[int] | None = None,
    ledger: TextIO | None = None,
    context: dict | None = None,
) -> tuple[dict, float]:
    params = {
        "temperature": 0,
        "sampling_seed": SEED,
        "ignore_eos": True,
        "max_new_tokens": output_length,
    }
    if replay is not None:
        params["replay_token_ids"] = replay
    payload = {"input_ids": input_ids, "sampling_params": params}
    wall_time_ns = time.time_ns()
    started = time.perf_counter()
    result = post(url, "/generate", payload)
    seconds = time.perf_counter() - started
    if ledger is not None:
        ledger.write(
            json.dumps(
                {
                    "wall_time_ns": wall_time_ns,
                    "context": context,
                    "request": payload,
                    "seconds": seconds,
                    "result": result,
                }
            )
            + "\n"
        )
        ledger.flush()
    return result, seconds

