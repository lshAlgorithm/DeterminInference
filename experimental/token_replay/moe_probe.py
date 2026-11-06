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


if __name__ == "__main__":
    raise SystemExit("use an explicit diagnostic entrypoint")
