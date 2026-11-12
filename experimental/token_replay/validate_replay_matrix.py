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


def run_cell(
    url: str,
    prompt: list[int],
    suffix: list[int],
    config: dict,
    cell: dict,
    ledger: TextIO,
    report: dict,
    output: Path,
) -> dict:
    width, concurrency = cell["output_length"], cell["concurrency"]
    label = f"l{width}-c{concurrency}"
    prompts = [prompt] * concurrency
    lengths = [width] * concurrency
    salts = [f"validation-{label}-{index}" for index in range(concurrency)]
    ordinary = payload_for(prompts, lengths, salts)
    post(url, "/flush_cache")
    reference, reference_seconds = issue(
        url, ordinary, ledger, {"cell": label, "phase": "reference"}
    )
    assert len(reference) == concurrency and all(
        len(row["output_ids"]) == width for row in reference
    )
    replay = payload_for(
        prompts, lengths, salts, [row["output_ids"] for row in reference]
    )
    post(url, "/flush_cache")
    first, first_seconds = issue(
        url, replay, ledger, {"cell": label, "phase": "first_replay"}
    )
    result = {
        "name": label,
        **cell,
        "first_reference_seconds": reference_seconds,
        "first_replay_seconds": first_seconds,
        "first_replay_check": check_rows(first, reference, prompts, lengths),
        "warmup_pairs": [],
        "pairs": [],
    }
    report["cells"].append(result)
    for index in range(-config["warmup_pairs"], config["pairs"]):
        pair = {"index": index, "arms": {}}
        order = ("ordinary", "replay") if index % 2 == 0 else ("replay", "ordinary")
        for mode in order:
            post(url, "/flush_cache")
            rows, seconds = issue(
                url,
                ordinary if mode == "ordinary" else replay,
                ledger,
                {
                    "cell": label,
                    "phase": "warmup" if index < 0 else "measured",
                    "pair": index,
                    "mode": mode,
                },
            )
            checks = check_rows(rows, reference, prompts, lengths)
            pair["arms"][mode] = {
                "seconds": seconds,
                **checks,
                "server_seconds": sum(row["meta_info"]["e2e_latency"] for row in rows)
                / concurrency,
            }
        pair["overhead_fraction"] = (
            pair["arms"]["replay"]["seconds"] / pair["arms"]["ordinary"]["seconds"] - 1
        )
        pair["valid"] = all(
            arm["ids_match"] and arm["work_match"] for arm in pair["arms"].values()
        )
        result["warmup_pairs" if index < 0 else "pairs"].append(pair)
        output.write_text(json.dumps(report, indent=2) + "\n")
    result["primary_metric"] = estimate(
        result["pairs"], config["critical"], config["target_fraction"]
    )
    result["cache_reuse"] = check_reuse(
        url, prompts, reference, salts, suffix, ledger, label
    )
    result["passed"] = (
        result["primary_metric"]["status"] == "pass"
        and all(result["first_replay_check"].values())
        and result["cache_reuse"]["passed"]
    )
    return result

