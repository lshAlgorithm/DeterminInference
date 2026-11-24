from __future__ import annotations

import json
import random
import statistics
from types import SimpleNamespace

import torch
from sglang.srt.sampling.token_replay import TokenReplayBatch


def profile() -> dict:
    rng = random.Random(42)
    torch.manual_seed(42)
    results = []
    for size in (1, 4, 8):
        requests = [
            SimpleNamespace(
                origin_input_ids=[1] * 32,
                sampling_params=SimpleNamespace(
                    replay_token_ids=[rng.randrange(151000) for _ in range(128)]
                ),
            )
            for _ in range(size)
        ]
        batch = TokenReplayBatch.from_requests(requests, "cuda")
        positions = torch.full((size,), 31, device="cuda", dtype=torch.int64)
        logits = torch.randn((size, 151000), device="cuda")
        prefetched = torch.empty((size,), device="cuda", dtype=torch.int32)
        for _ in range(5):
            sampled = torch.argmax(logits, dim=-1).to(torch.int32)
            batch.apply(sampled, positions)
        torch.cuda.synchronize()

        baseline_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(baseline_graph):
            baseline = torch.argmax(logits, dim=-1).to(torch.int32)
            batch.apply(baseline, positions)

        prefetch_stream = torch.cuda.Stream()
        current_stream = torch.cuda.current_stream()
        candidate_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(candidate_graph):
            with torch.cuda.stream(prefetch_stream):
                batch.apply(prefetched, positions)
            candidate = torch.argmax(logits, dim=-1).to(torch.int32)
            current_stream.wait_stream(prefetch_stream)
            candidate.copy_(prefetched)

        expected = [request.sampling_params.replay_token_ids[0] for request in requests]
        for graph in (baseline_graph, candidate_graph):
            for _ in range(5):
                graph.replay()
            torch.cuda.synchronize()
        assert baseline.tolist() == candidate.tolist() == expected

        pairs = []
        for repeat in range(30):
            order = (
                (baseline_graph, candidate_graph)
                if repeat % 2 == 0
                else (candidate_graph, baseline_graph)
            )
            durations = []
            for graph in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                durations.append(start.elapsed_time(end) * 1000)
            arms = dict(
                zip(
                    ("baseline", "candidate")
                    if repeat % 2 == 0
                    else ("candidate", "baseline"),
                    durations,
                    strict=True,
                )
            )
            pairs.append(arms)

        base = statistics.median(pair["baseline"] for pair in pairs)
        candidate_time = statistics.median(pair["candidate"] for pair in pairs)
        results.append(
            {
                "batch_size": size,
                "vocabulary_size": logits.shape[-1],
                "pairs": pairs,
                "baseline_device_us_median": base,
                "async_prefetch_device_us_median": candidate_time,
                "candidate_relative_change": candidate_time / base - 1,
                "token_equality": "passed",
                "scope": "argmax surrogate; CUDA graph; side-stream prefetch plus post-sample copy",
            }
        )
    return {
        "seed": 42,
        "gpu": torch.cuda.get_device_name(),
        "method": "30 alternating paired graph replays per batch; five warmups per graph",
        "results": results,
    }


if __name__ == "__main__":
    print(json.dumps(profile(), indent=2))
