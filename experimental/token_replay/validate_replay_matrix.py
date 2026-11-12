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


def estimate(pairs: list[dict], critical: float, target: float) -> dict:
    values = [pair["overhead_fraction"] for pair in pairs]
    mean = statistics.mean(values)
    half = critical * statistics.stdev(values) / math.sqrt(len(values))
    valid = all(pair["valid"] for pair in pairs)
    return {
        "name": "replay_http_wall_time_overhead",
        "unit": "fraction",
        "definition": "replay batch wall time / ordinary batch wall time - 1",
        "mean": mean,
        "interval": [mean - half, mean + half],
        "critical": critical,
        "valid": valid,
        "target_fraction": target,
        "min_pair": min(values),
        "max_pair": max(values),
        "status": (
            "invalid"
            if not valid
            else "pass"
            if mean + half <= target
            else "fail"
            if mean - half > target
            else "inconclusive"
        ),
        "ordinary_id_mismatch_pairs": sum(
            not p["arms"]["ordinary"]["ids_match"] for p in pairs
        ),
        "replay_id_mismatch_pairs": sum(
            not p["arms"]["replay"]["ids_match"] for p in pairs
        ),
    }


def check_reuse(
    url: str,
    prompts: list[list[int]],
    reference: list[dict],
    salts: list[str],
    suffix: list[int],
    ledger: TextIO,
    label: str,
) -> dict:
    recorded = [row["output_ids"] for row in reference]
    lengths = list(map(len, recorded))
    post(url, "/flush_cache")
    primed, _ = issue(
        url,
        payload_for(prompts, lengths, salts, recorded),
        ledger,
        {"cell": label, "phase": "cache_prime"},
    )
    assert [row["output_ids"] for row in primed] == recorded
    continuation = [
        prompt + output + suffix
        for prompt, output in zip(prompts, recorded, strict=True)
    ]
    followup = payload_for(continuation, [8] * len(prompts), salts)
    cached, _ = issue(url, followup, ledger, {"cell": label, "phase": "cache_hit"})
    post(url, "/flush_cache")
    cold, _ = issue(url, followup, ledger, {"cell": label, "phase": "cache_cold"})
    eligible = [
        len(prompt) + len(output) - 1
        for prompt, output in zip(prompts, recorded, strict=True)
    ]
    hits = [row["meta_info"]["cached_tokens"] for row in cached]
    assert all(row["meta_info"]["cached_tokens"] == 0 for row in cold)
    assert all(hit >= count for hit, count in zip(hits, eligible, strict=True)), (
        hits,
        eligible,
    )
    assert [row["output_ids"] for row in cached] == [row["output_ids"] for row in cold]
    assert all(row["meta_info"]["total_retractions"] == 0 for row in cached + cold)
    return {
        "passed": True,
        "eligible_prefix_tokens": eligible,
        "cached_tokens": hits,
        "eligible_prefix_reuse_fraction": [
            min(hit / count, 1.0) for hit, count in zip(hits, eligible, strict=True)
        ],
    }

