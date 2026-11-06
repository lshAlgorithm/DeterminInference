from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
import urllib.request
from pathlib import Path


def request_json(url: str, endpoint: str, payload: dict | None = None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url + endpoint, data, {'Content-Type':'application/json'})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)
