from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import time
from pathlib import Path


def guard_gpus() -> list[dict]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
        raise ValueError("this diagnostic requires CUDA_VISIBLE_DEVICES=2,3")
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    devices = []
    for row in rows.splitlines():
        index, name, used, total, utilization = [part.strip() for part in row.split(",")]
        if index in {"2", "3"}:
            device = dict(index=int(index), name=name, used_mib=int(used), total_mib=int(total), utilization=int(utilization))
            if device["used_mib"] >= 500:
                raise RuntimeError(f"GPU {index} is occupied: {device}")
            devices.append(device)
    if len(devices) != 2:
        raise RuntimeError("both physical GPUs 2 and 3 must be visible")
    return devices


def model_manifest(model: Path) -> dict:
    index = json.loads((model / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    files = []
    for name in ["config.json", "tokenizer_config.json", "model.safetensors.index.json", *shards]:
        path = model / name
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(8 << 20):
                digest.update(chunk)
        files.append(dict(name=name, size_bytes=path.stat().st_size, sha256=digest.hexdigest()))
    return dict(
        model="Qwen/Qwen1.5-MoE-A2.7B-Chat",
        revision="ec052fda178e241c7c443468d2fa1db6618996be",
        declared_weight_bytes=index["metadata"]["total_size"],
        files=files,
    )


if __name__ == "__main__":
    raise SystemExit("use an explicit diagnostic entrypoint")
