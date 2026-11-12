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


def test_exact_output(
    url: str, prompt_ids: list[int], ledger: TextIO | None = None
) -> tuple[list[int], dict]:
    post(url, "/flush_cache")
    ordinary, _ = generate(
        url, prompt_ids, 32, ledger=ledger, context={"phase": "reference"}
    )
    recorded = ordinary["output_ids"]
    assert len(recorded) == 32
    post(url, "/flush_cache")
    replayed, _ = generate(
        url,
        prompt_ids,
        len(recorded),
        recorded,
        ledger,
        {"phase": "exact_output", "mode": "replay"},
    )
    assert replayed["output_ids"] == recorded, replayed
    assert replayed["text"] == ordinary["text"]
    assert replayed["meta_info"]["finish_reason"]["type"] == "length"

    return recorded, {"passed": True, "output_tokens": len(recorded)}


def measure_pairs(
    url: str,
    prompt_ids: list[int],
    recorded: list[int],
    pair_count: int,
    requests_per_arm: int,
    warmup_pairs: int,
    ledger: TextIO,
    report: dict,
    output: Path,
) -> None:
    pairs = report["primary_metric"]["pairs"]
    for index in range(-warmup_pairs, pair_count):
        pair = {"index": index, "arms": {}}
        order = ("ordinary", "replay") if index % 2 == 0 else ("replay", "ordinary")
        for mode in order:
            requests = []
            for request_index in range(requests_per_arm):
                post(url, "/flush_cache")
                result, seconds = generate(
                    url,
                    prompt_ids,
                    len(recorded),
                    recorded if mode == "replay" else None,
                    ledger,
                    {
                        "phase": "warmup" if index < 0 else "measured",
                        "pair": index,
                        "mode": mode,
                        "request": request_index,
                    },
                )
                meta = result["meta_info"]
                requests.append(
                    {
                        "seconds": seconds,
                        "server_seconds": meta.get("e2e_latency"),
                        "output_ids_match": result["output_ids"] == recorded,
                        "work_matches": (
                            meta["cached_tokens"] == 0
                            and meta["prompt_tokens"] == len(prompt_ids)
                            and meta["completion_tokens"] == len(recorded)
                            and meta["finish_reason"]["type"] == "length"
                            and meta["total_retractions"] == 0
                        ),
                    }
                )
            server_times = [request["server_seconds"] for request in requests]
            pair["arms"][mode] = {
                "seconds": sum(request["seconds"] for request in requests),
                "server_seconds": (
                    sum(server_times)
                    if all(value is not None for value in server_times)
                    else None
                ),
                "requests": requests,
            }
        ordinary, replay = pair["arms"]["ordinary"], pair["arms"]["replay"]
        pair["overhead_fraction"] = replay["seconds"] / ordinary["seconds"] - 1
        pair["server_overhead_fraction"] = (
            replay["server_seconds"] / ordinary["server_seconds"] - 1
            if ordinary["server_seconds"] and replay["server_seconds"]
            else None
        )
        pair["valid"] = all(
            request["output_ids_match"] and request["work_matches"]
            for arm in pair["arms"].values()
            for request in arm["requests"]
        )
        (report["warmup_pairs"] if index < 0 else pairs).append(pair)
        output.write_text(json.dumps(report, indent=2) + "\n")


def summarize_overhead(metric: dict) -> None:
    values = [pair["overhead_fraction"] for pair in metric["pairs"]]
    mean = statistics.mean(values)
    critical = {5: 2.776445105, 20: 2.093024054}[len(values)]
    half = critical * statistics.stdev(values) / math.sqrt(len(values))
    valid = all(pair["valid"] for pair in metric["pairs"])
    metric.update(
        {
            "mean": mean,
            "ci95": [mean - half, mean + half],
            "method": (
                "fixed final paired Student t interval under t assumptions; "
                "no sample filtering or early stopping"
            ),
            "median_pair": statistics.median(values),
            "min_pair": min(values),
            "max_pair": max(values),
            "valid": valid,
            "target_fraction": 0.03,
            "status": (
                "invalid"
                if not valid
                else "pass"
                if mean + half <= 0.03
                else "fail"
                if mean - half > 0.03
                else "inconclusive"
            ),
        }
    )


def test_roomy_cache_reuse(
    url: str, prompt_ids: list[int], recorded: list[int], ledger: TextIO | None = None
) -> dict:
    suffix_ids = post(
        url, "/tokenize", {"prompt": SUFFIX, "add_special_tokens": False}
    )["tokens"]
    followup = prompt_ids + recorded + suffix_ids
    post(url, "/flush_cache")
    primed, _ = generate(
        url, prompt_ids, len(recorded), recorded, ledger, {"phase": "cache_prime"}
    )
    assert primed["output_ids"] == recorded
    cached, _ = generate(
        url, followup, 8, ledger=ledger, context={"phase": "cache_hit"}
    )
    post(url, "/flush_cache")
    cold, _ = generate(url, followup, 8, ledger=ledger, context={"phase": "cache_cold"})

    eligible_prefix = len(prompt_ids) + len(recorded) - 1
    reused = cached["meta_info"]["cached_tokens"]
    assert cold["meta_info"]["cached_tokens"] == 0, cold
    assert reused >= eligible_prefix, (reused, eligible_prefix)
    assert cached["output_ids"] == cold["output_ids"]
    return {
        "passed": True,
        "eligible_prefix_tokens": eligible_prefix,
        "cached_tokens": reused,
        "eligible_prefix_reuse_fraction": min(reused / eligible_prefix, 1.0),
        "followup_output_tokens": len(cached["output_ids"]),
    }

