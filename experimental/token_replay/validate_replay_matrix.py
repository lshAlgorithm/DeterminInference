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


def check_rows(
    rows: list[dict],
    reference: list[dict],
    prompts: list[list[int]],
    lengths: list[int],
) -> dict:
    size_ok = len(rows) == len(reference)
    ids_ok = size_ok and all(
        row["output_ids"] == expected["output_ids"] and row["text"] == expected["text"]
        for row, expected in zip(rows, reference, strict=True)
    )
    work_ok = size_ok and all(
        row["meta_info"]["cached_tokens"] == 0
        and row["meta_info"]["prompt_tokens"] == len(prompt)
        and row["meta_info"]["completion_tokens"] == length
        and row["meta_info"]["total_retractions"] == 0
        and row["meta_info"]["finish_reason"]["type"] == "length"
        for row, prompt, length in zip(rows, prompts, lengths, strict=True)
    )
    return {"ids_match": ids_ok, "work_match": work_ok}

