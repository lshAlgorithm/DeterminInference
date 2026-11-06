from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

from run_campaign import gpu_guard


def measure_storage(source: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    guard = gpu_guard([2])
    import torch
    from sglang.srt.sampling.token_replay import TokenReplayBatch

    rows = [(p.name, i, row) for p in sorted(source.glob('*.json')) for i, row in enumerate(json.loads(p.read_text())['requests'])]
    selected = random.Random(42).sample(rows, 32)
    torch.manual_seed(42)
    started = time.perf_counter()
    warm = torch.empty(1, device='cuda')
    torch.cuda.synchronize()
    initialization = time.perf_counter()-started
    del warm
    records = []
    for concurrency in [1, 4, 8]:
        for offset in range(0, len(selected), concurrency):
            group = selected[offset:offset+concurrency]
            requests = [SimpleNamespace(origin_input_ids=row['input_ids'], sampling_params=SimpleNamespace(replay_token_ids=row['output_ids'])) for _, _, row in group]
            keep = list(range(max(1, concurrency//2)))
            indices = torch.tensor(keep, device='cuda')
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            baseline = torch.cuda.memory_allocated()
            reserved_baseline = torch.cuda.memory_reserved()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            batch = TokenReplayBatch.from_requests(requests, 'cuda')
            torch.cuda.synchronize()
            staging_seconds = time.perf_counter()-started
            staged_bytes = torch.cuda.memory_allocated()-baseline
            started = time.perf_counter()
            filtered = batch.filter(indices, keep)
            torch.cuda.synchronize()
            filter_seconds = time.perf_counter()-started
            del batch
            torch.cuda.synchronize()
            filtered_bytes = torch.cuda.memory_allocated()-baseline
            started = time.perf_counter()
            merged = TokenReplayBatch.merge(filtered, None, len(keep), concurrency, 'cuda')
            torch.cuda.synchronize()
            merge_seconds = time.perf_counter()-started
            del filtered
            torch.cuda.synchronize()
            merged_bytes = torch.cuda.memory_allocated()-baseline
            peak_bytes = torch.cuda.max_memory_allocated()-baseline
            reserved_peak_delta = torch.cuda.max_memory_reserved()-reserved_baseline
            del merged
            gc.collect()
            torch.cuda.synchronize()
            released_bytes = torch.cuda.memory_allocated()-baseline
            assert released_bytes == 0, released_bytes
            record = {'concurrency': concurrency, 'sources': [[name, index] for name, index, _ in group],
                      'baseline_allocated_bytes': baseline, 'baseline_reserved_bytes': reserved_baseline,
                      'token_payload_bytes': 8*sum(len(row['output_ids']) for _, _, row in group),
                      'metadata_payload_bytes': 24*concurrency,
                      'active_payload_after_filter_bytes': 8*sum(len(group[i][2]['output_ids']) for i in keep)+24*len(keep),
                      'staged_allocated_bytes': staged_bytes, 'filtered_allocated_bytes': filtered_bytes,
                      'merged_allocated_bytes': merged_bytes, 'peak_allocated_bytes': peak_bytes,
                      'peak_reserved_delta_bytes': reserved_peak_delta, 'allocated_after_release_delta_bytes': released_bytes,
                      'staging_seconds': staging_seconds, 'filter_seconds': filter_seconds, 'merge_seconds': merge_seconds}
            records.append(record)
            del indices
    result = {'complete': True, 'seed': 42, 'gpu_guard': guard, 'gpu': torch.cuda.get_device_name(),
              'source': str(source), 'cuda_first_allocation_seconds': initialization, 'records': records,
              'scope': 'standalone replay-storage component on32 fixed qualified responses; preallocated filter indices excluded; real CUDA allocator allocation/peak deltas, not whole-serving incremental VRAM; filter retains shared packed storage; merge adds a same-concurrency ordinary batch; release checked after synchronize; reserved pool retention is distinct from live allocation; timings are component diagnostics, not the3% end-to-end decision'}
    (output/'storage.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    measure_storage(args.source, args.output)
