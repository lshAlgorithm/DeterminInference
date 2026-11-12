from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.request
from pathlib import Path
from typing import TextIO


PROMPT = (
    "Explain why a clear sky appears blue in two short sentences. "
    "Use ordinary words and avoid lists. "
) * 12
SUFFIX = "\nNow summarize the answer in one sentence."
SEED = 42


def post(url: str, path: str, payload: dict | None = None) -> dict:
    request = urllib.request.Request(
        url.rstrip("/") + path,
        json.dumps(payload).encode() if payload is not None else b"{}",
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        if path == "/flush_cache":
            body = response.read().decode()
            assert response.status == 200 and body.startswith("Cache flushed."), body
            return {}
        return json.load(response)

