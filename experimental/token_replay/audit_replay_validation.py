from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def audit_cell(records: list[dict], cell: dict, config: dict) -> dict:
    label = cell["name"]
    rows = [row for row in records if row["context"].get("cell") == label]
    assert not any("error" in row for row in rows), label
    reference = next(row for row in rows if row["context"]["phase"] == "reference")
    expected = reference["result"]
    concurrency = cell["concurrency"]
    width = cell["output_length"]
    assert len(expected) == concurrency
    salts = reference["request"]["extra_key"]
    assert len(set(salts)) == concurrency
    observations = {}
    for row in rows:
        phase = row["context"]["phase"]
        if phase not in ("reference", "first_replay", "warmup", "measured"):
            continue
        request, response = row["request"], row["result"]
        assert len(response) == concurrency
        assert request["input_ids"] == reference["request"]["input_ids"]
        assert request["extra_key"] == salts
        replay = phase == "first_replay" or row["context"].get("mode") == "replay"
        for index, (actual, target) in enumerate(zip(response, expected, strict=True)):
            params = request["sampling_params"][index]
            assert params["sampling_seed"] == 42 and params["temperature"] == 0
            assert params["ignore_eos"] and params["max_new_tokens"] == width
            assert ("replay_token_ids" in params) == replay
            if replay:
                assert params["replay_token_ids"] == target["output_ids"]
            assert actual["output_ids"] == target["output_ids"]
            assert actual["text"] == target["text"]
            meta = actual["meta_info"]
            assert len(actual["output_ids"]) == meta["completion_tokens"] == width
            assert meta["cached_tokens"] == meta["total_retractions"] == 0
            assert meta["prompt_tokens"] == len(request["input_ids"][index])
            assert meta["finish_reason"]["type"] == "length"
        if phase in ("warmup", "measured"):
            key = (row["context"]["pair"], row["context"]["mode"])
            assert key not in observations
            observations[key] = row["seconds"]
    assert len(observations) == 2 * (config["pairs"] + config["warmup_pairs"])
    values = []
    for index in range(-config["warmup_pairs"], config["pairs"]):
        value = observations[index, "replay"] / observations[index, "ordinary"] - 1
        reported = cell["warmup_pairs" if index < 0 else "pairs"][
            index + config["warmup_pairs"] if index < 0 else index
        ]
        assert reported["index"] == index and reported["valid"]
        assert math.isclose(value, reported["overhead_fraction"], abs_tol=1e-12)
        if index >= 0:
            values.append(value)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    half = config["critical"] * math.sqrt(variance / len(values))
    interval = [mean - half, mean + half]
    metric = cell["primary_metric"]
    assert math.isclose(mean, metric["mean"], abs_tol=1e-12)
    assert all(
        math.isclose(a, b, abs_tol=1e-12)
        for a, b in zip(interval, metric["interval"], strict=True)
    )
    cache = {
        phase: next(row for row in rows if row["context"]["phase"] == phase)
        for phase in ("cache_prime", "cache_hit", "cache_cold")
    }
    assert [row["output_ids"] for row in cache["cache_prime"]["result"]] == [
        row["output_ids"] for row in expected
    ]
    assert cache["cache_hit"]["request"] == cache["cache_cold"]["request"]
    reuse = []
    for index in range(concurrency):
        hit = cache["cache_hit"]["result"][index]
        cold = cache["cache_cold"]["result"][index]
        followup = cache["cache_hit"]["request"]["input_ids"][index]
        prefix = (
            reference["request"]["input_ids"][index] + expected[index]["output_ids"]
        )
        assert followup[: len(prefix)] == prefix
        assert hit["output_ids"] == cold["output_ids"] and hit["text"] == cold["text"]
        assert hit["meta_info"]["cached_tokens"] >= len(prefix) - 1
        assert cold["meta_info"]["cached_tokens"] == 0
        assert hit["meta_info"]["total_retractions"] == 0
        assert cold["meta_info"]["total_retractions"] == 0
        reuse.append(min(hit["meta_info"]["cached_tokens"] / (len(prefix) - 1), 1))
    return {
        "cell": label,
        "measured_requests": 2 * len(values) * concurrency,
        "measured_output_tokens": 2 * len(values) * concurrency * width,
        "mean": mean,
        "interval": interval,
        "overhead_target_passed": interval[1] <= config["target_fraction"],
        "eligible_prefix_reuse_fraction": reuse,
        "token_and_work_checks": "passed",
    }

