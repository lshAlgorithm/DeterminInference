from __future__ import annotations

import argparse
import hashlib
import json
import queue
import random
import tarfile
import threading
import time
import urllib.request
from pathlib import Path


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
