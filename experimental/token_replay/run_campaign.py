from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path


def gpu_guard(indices: list[int]) -> str:
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    found = set()
    for line in result.splitlines():
        index, used, _, _ = map(int, line.split(','))
        if index in indices:
            found.add(index)
            if used >= 500:
                raise RuntimeError(f'GPU {index} occupied: {used} MiB')
    if found != set(indices):
        raise RuntimeError(f'missing GPUs: {indices}, found {found}')
    return result


def stop_process(process: subprocess.Popen | None) -> None:
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
