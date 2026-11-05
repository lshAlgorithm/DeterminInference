import json
import random
import statistics
import time
from types import SimpleNamespace

import torch
from sglang.srt.sampling.token_replay import TokenReplayBatch


def profile_replay() -> dict:
    rng = random.Random(42)
    torch.manual_seed(42)
    results = []
    for size, width in ((1, 128), (32, 128), (32, 8192), (128, 8192), (128, 32768)):
        requests = [
            SimpleNamespace(
                origin_input_ids=[1] * 32,
                sampling_params=SimpleNamespace(
                    replay_token_ids=[rng.randrange(151000) for _ in range(width if i == 0 else 128)]
                    if i % 4 != 3 else None
                ),
            )
            for i in range(size)
        ]
        torch.cuda.synchronize()
        start = time.perf_counter()
        batch = TokenReplayBatch.from_requests(requests, 'cuda')
        torch.cuda.synchronize()
        staging_ms = (time.perf_counter() - start) * 1000
        keep_indices = list(range(size - 1, -1, -1))
        indices = torch.tensor(keep_indices, device='cuda')
        sampled = torch.full((size,), -2, device='cuda', dtype=torch.int32)
        positions = torch.full((size,), 31, device='cuda', dtype=torch.int64)
        batch.apply(sampled, positions)
        expected = [r.sampling_params.replay_token_ids[0] if r.sampling_params.replay_token_ids is not None else -2 for r in requests]
        assert sampled.tolist() == expected
        timings = {}
        for operation in ('apply', 'filter', 'merge'):
            measurements = []
            for iteration in range(12):
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(20):
                    if operation == 'apply':
                        batch.apply(sampled, positions)
                    elif operation == 'filter':
                        filtered = batch.filter(indices, keep_indices)
                    else:
                        merged = TokenReplayBatch.merge(batch, None, size, size, 'cuda')
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1e6 / 20
                if iteration >= 2:
                    measurements.append(elapsed)
            timings[operation] = {'median_us': statistics.median(measurements), 'samples_us': measurements}
        results.append({
            'batch_size': size, 'max_response': width,
            'real_token_count': sum(len(r.sampling_params.replay_token_ids or []) for r in requests),
            'token_table_bytes': sum(t.numel() * t.element_size() for t in batch.tokens if t is not None),
            'staging_ms': staging_ms, 'operations': timings,
        })
    return {'seed': 42, 'gpu': torch.cuda.get_device_name(), 'timing': 'synchronized host wall time, 20 calls/sample, 2 warmup + 10 measured samples', 'results': results}


if __name__ == '__main__':
    print(json.dumps(profile_replay(), indent=2))
