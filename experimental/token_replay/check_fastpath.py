import json
from types import SimpleNamespace

import torch
from sglang.srt.sampling.token_replay import TokenReplayBatch


def check_fastpath() -> dict:
    requests = [SimpleNamespace(origin_input_ids=[1], sampling_params=SimpleNamespace(replay_token_ids=ids)) for ids in ([7, 8], None, None)]
    original = TokenReplayBatch.from_requests(requests, 'cuda')
    ordinary = original.filter(torch.tensor([2, 1], device='cuda'), [2, 1])
    assert ordinary is None, 'ordinary-only batch still dispatches replay kernel'
    assert original.filter(torch.empty(0, device='cuda', dtype=torch.int64), []) is None
    sampled = torch.tensor([90, 91, 92], device='cuda')
    original.apply(sampled, torch.zeros(3, device='cuda', dtype=torch.int64))
    assert sampled.tolist() == [7, 91, 92], 'in-flight batch was mutated'
    active = TokenReplayBatch.from_requests(requests[:1], 'cuda')
    rejoined = TokenReplayBatch.merge(ordinary, active, 2, 1, 'cuda')
    sampled = torch.tensor([90, 91, 92], device='cuda')
    rejoined.apply(sampled, torch.zeros(3, device='cuda', dtype=torch.int64))
    assert sampled.tolist() == [90, 91, 7]
    return {'ordinary_fastpath': 'passed', 'inflight_and_rejoin': 'passed'}


if __name__ == '__main__':
    print(json.dumps(check_fastpath(), indent=2))
