from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from infx.bench.env import InputError


def load_sessions(path: Path) -> dict[str, list[dict]]:
    """Load ordered, full-token turns grouped by independent session."""
    sessions: dict[str, list[dict]] = {}
    with path.open() as source:
        for number, line in enumerate(source, 1):
            row = json.loads(line)
            required = {"session_id", "input_ids", "output_ids", "delay_seconds"}
            if not isinstance(row, dict) or set(row) != required:
                raise InputError(f"line {number}: expected exactly {sorted(required)}")
            if not isinstance(row["session_id"], str) or not row["session_id"]:
                raise InputError(f"line {number}: session_id must be a nonempty string")
            for field in ("input_ids", "output_ids"):
                ids = row[field]
                if not isinstance(ids, list) or not ids or any(type(t) is not int or t < 0 for t in ids):
                    raise InputError(f"line {number}: {field} must contain nonnegative integer IDs")
            delay = row["delay_seconds"]
            if type(delay) not in (int, float) or not math.isfinite(delay) or delay < 0:
                raise InputError(f"line {number}: delay_seconds must be finite and nonnegative")
            sessions.setdefault(row["session_id"], []).append(row)
    if not sessions:
        raise InputError("the trace contains no sessions")
    return sessions
