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

