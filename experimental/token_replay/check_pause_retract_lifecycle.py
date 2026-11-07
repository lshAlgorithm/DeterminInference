from __future__ import annotations

import argparse
import json
import queue
import time
import urllib.request
from pathlib import Path

from check_retraction_lifecycle import (
    abort_owned_request,
    canonical_hash,
    launch_request,
    make_request,
    read_observer_events,
    receive_message,
    validate_request_events,
)


def control(url: str, endpoint: str, payload: dict[str, str]) -> dict:
    request = urllib.request.Request(
        url.rstrip("/") + endpoint,
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        result = {"status": response.status, "body": json.load(response)}
    if result["status"] != 200 or result["body"].get("status") != "ok":
        raise ValueError(f"control request failed: {result}")
    return result
