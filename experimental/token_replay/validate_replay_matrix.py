from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import TextIO

from check_text_replay_smoke import PROMPT, SEED, SUFFIX, post


def issue(
    url: str, payload: dict, ledger: TextIO, context: dict
) -> tuple[list[dict], float]:
    wall_time_ns = time.time_ns()
    started = time.perf_counter()
    try:
        result = post(url, "/generate", payload)
    except Exception as error:
        ledger.write(
            json.dumps(
                {
                    "context": context,
                    "wall_time_ns": wall_time_ns,
                    "request": payload,
                    "error": repr(error),
                }
            )
            + "\n"
        )
        ledger.flush()
        raise
    seconds = time.perf_counter() - started
    rows = result if isinstance(result, list) else [result]
    ledger.write(
        json.dumps(
            {
                "context": context,
                "wall_time_ns": wall_time_ns,
                "request": payload,
                "result": rows,
                "seconds": seconds,
            }
        )
        + "\n"
    )
    ledger.flush()
    return rows, seconds


def payload_for(
    prompts: list[list[int]],
    lengths: list[int],
    salts: list[str],
    recorded: list[list[int] | None] | None = None,
) -> dict:
    params = []
    for index, length in enumerate(lengths):
        item = {
            "temperature": 0,
            "sampling_seed": SEED,
            "ignore_eos": True,
            "max_new_tokens": length,
        }
        if recorded is not None and recorded[index] is not None:
            item["replay_token_ids"] = recorded[index]
        params.append(item)
    return {"input_ids": prompts, "sampling_params": params, "extra_key": salts}

