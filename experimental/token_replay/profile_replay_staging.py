from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.sampling import token_replay


class TensorFactory:
    def __init__(self, pinned: bool):
        self.pinned = pinned

    def __getattr__(self, name):
        return getattr(torch, name)

    def tensor(self, data, **kwargs):
        device = kwargs.get("device")
        if self.pinned and device is not None and torch.device(device).type == "cuda":
            host = torch.tensor(data, **{**kwargs, "device": "cpu", "pin_memory": True})
            return host.to(device, non_blocking=True)
        return torch.tensor(data, **kwargs)


def stage(requests: list, mode: str):
    with patch.object(token_replay, "torch", TensorFactory(mode == "pinned_async")):
        return token_replay.TokenReplayBatch.from_requests(requests, "cuda")


def validate(batch, requests: list) -> None:
    gc.collect()
    expected = [request.sampling_params.replay_token_ids for request in requests]
    assert [
        token.tolist() if token is not None else None for token in batch.tokens
    ] == expected
    sampled = torch.full((len(requests),), -1, device="cuda", dtype=torch.int32)
    positions = torch.full((len(requests),), 31, device="cuda", dtype=torch.int64)
    batch.apply(sampled, positions)
    assert sampled.tolist() == [
        sequence[0] if sequence is not None else -1 for sequence in expected
    ]


def profile(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    rng = random.Random(42)
    report = {
        "seed": 42,
        "gpu": torch.cuda.get_device_name(),
        "scope": (
            "constructor candidate only; pinned allocation and H2D included; "
            "both modes include the same patch dispatch; no serving claim"
        ),
        "cases": [],
    }
    for size, width in ((1, 32), (1, 128), (32, 128), (128, 128), (32, 8192)):
        requests = [
            SimpleNamespace(
                origin_input_ids=[1] * 32,
                sampling_params=SimpleNamespace(
                    replay_token_ids=(
                        [rng.randrange(151000) for _ in range(width)]
                        if row % 4 != 3
                        else None
                    )
                ),
            )
            for row in range(size)
        ]
        for _ in range(5):
            for mode in ("baseline", "pinned_async"):
                validate(stage(requests, mode), requests)
        pairs = []
        for repeat in range(30):
            pair = {}
            order = (
                ("baseline", "pinned_async")
                if repeat % 2 == 0
                else ("pinned_async", "baseline")
            )
            for mode in order:
                torch.cuda.synchronize()
                started = time.perf_counter()
                batch = stage(requests, mode)
                host_us = (time.perf_counter() - started) * 1e6
                torch.cuda.synchronize()
                ready_us = (time.perf_counter() - started) * 1e6
                pair[mode] = {"host_return_us": host_us, "ready_us": ready_us}
                if repeat == 29:
                    validate(batch, requests)
            pairs.append(pair)
        case = {"batch_size": size, "response_length": width, "pairs": pairs}
        for metric in ("host_return_us", "ready_us"):
            medians = {
                mode: statistics.median(pair[mode][metric] for pair in pairs)
                for mode in ("baseline", "pinned_async")
            }
            case[metric] = {
                **medians,
                "candidate_relative_change": medians["pinned_async"]
                / medians["baseline"]
                - 1,
            }
        report["cases"].append(case)
        output.write_text(json.dumps(report, indent=2) + "\n")

    report["profiler"] = {}
    for mode in ("baseline", "pinned_async"):
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            with torch.profiler.record_function("replay_staging_" + mode):
                batch = stage(requests, mode)
            torch.cuda.synchronize()
        profiler.export_chrome_trace(
            str(output.with_name(mode + "-staging-trace.json"))
        )
        report["profiler"][mode] = [
            {
                "name": item.key,
                "count": item.count,
                "self_cpu_us": item.self_cpu_time_total,
            }
            for item in profiler.key_averages()
            if "Memcpy" in item.key
            or "Synchronize" in item.key
            or "aten::to" == item.key
        ]
    report["complete"] = True
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "complete": True,
                "cases": [
                    {key: value for key, value in case.items() if key != "pairs"}
                    for case in report["cases"]
                ],
                "profiler": report["profiler"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    profile(parser.parse_args().output)
