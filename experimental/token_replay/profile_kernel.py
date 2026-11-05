import importlib
import json
import os
import random
import statistics
from types import SimpleNamespace

import torch


def profile_kernel() -> dict:
    module = importlib.import_module(os.environ['REPLAY_PROFILE_MODULE'])
    rng = random.Random(42)
    results = []
    for size in (1, 32, 128, 512):
        requests = [SimpleNamespace(
            origin_input_ids=[1] * 32,
            sampling_params=SimpleNamespace(replay_token_ids=[rng.randrange(151000) for _ in range(128)] if i % 4 != 3 else None),
        ) for i in range(size)]
        batch = module.TokenReplayBatch.from_requests(requests, 'cuda')
        sampled = torch.full((size,), -1, device='cuda', dtype=torch.int32)
        positions = torch.full((size,), 31, device='cuda', dtype=torch.int64)
        batch.apply(sampled, positions)
        assert sampled.tolist() == [r.sampling_params.replay_token_ids[0] if r.sampling_params.replay_token_ids else -1 for r in requests]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(5):
                batch.apply(sampled, positions)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(100):
                batch.apply(sampled, positions)
        samples = []
        for index in range(12):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            if index >= 2:
                samples.append(start.elapsed_time(end) * 1000 / 100)
        results.append({'batch_size': size, 'median_device_us': statistics.median(samples), 'samples_device_us': samples})
    return {'seed': 42, 'module': module.__name__, 'gpu': torch.cuda.get_device_name(), 'method': 'CUDA events around graph with 100 replacement kernels; 2 warmup and 10 measured graph replays', 'results': results}


if __name__ == '__main__':
    print(json.dumps(profile_kernel(), indent=2))
