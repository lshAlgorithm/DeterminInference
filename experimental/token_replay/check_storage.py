import gc
import json
import random
import weakref
from types import SimpleNamespace

import torch
from sglang.srt.sampling.token_replay import TokenReplayBatch


def check_storage() -> dict:
    rng = random.Random(42)
    requests = [SimpleNamespace(origin_input_ids=[1] * 4, sampling_params=SimpleNamespace(replay_token_ids=[11, 12]))]
    batch = TokenReplayBatch.from_requests(requests, 'cuda')
    owner = weakref.ref(batch.tokens[0]._base)
    empty = batch.filter(torch.empty(0, device='cuda', dtype=torch.int64), [])
    torch.cuda.synchronize()
    del batch
    gc.collect()
    assert owner() is None, 'filtered-out admission storage was retained'
    batch = empty
    expected = []
    for iteration in range(200):
        incoming = []
        for _ in range(rng.randrange(1, 5)):
            prompt = rng.randrange(1, 20)
            sequence = [rng.randrange(151000) for _ in range(rng.randrange(1, 33))] if rng.random() < .75 else None
            incoming.append(SimpleNamespace(origin_input_ids=[1] * prompt, sampling_params=SimpleNamespace(replay_token_ids=sequence)))
            expected.append((prompt, sequence))
        other = TokenReplayBatch.from_requests(incoming, 'cuda')
        batch = TokenReplayBatch.merge(batch, other, len(expected) - len(incoming), len(incoming), 'cuda')
        del other
        keep = rng.sample(range(len(expected)), min(16, len(expected)))
        if batch is not None:
            batch = batch.filter(torch.tensor(keep, device='cuda'), keep)
        expected = [expected[i] for i in keep]
        positions, outputs = [], []
        for prompt, sequence in expected:
            step = rng.randrange(-1, (len(sequence) if sequence else 0) + 2)
            positions.append(prompt - 1 + step)
            outputs.append(sequence[step] if sequence is not None and 0 <= step < len(sequence) else -7)
        sampled = torch.full((len(expected),), -7, device='cuda', dtype=torch.int32 if iteration % 2 else torch.int64)
        if batch is not None:
            batch.apply(sampled, torch.tensor(positions, device='cuda'))
        assert sampled.tolist() == outputs, iteration
        if iteration % 20 == 0:
            gc.collect()
    return {'seed': 42, 'rounds': 200, 'reorder_merge_bounds_lifetime': 'passed', 'released_admission_storage': 'passed'}


if __name__ == '__main__':
    print(json.dumps(check_storage(), indent=2))
