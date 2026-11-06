from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

DATASET = "Exgentic/agent-llm-traces-v2"
REVISION = "4b8ad4ab198438e5a170f9171c19c6a2cf7c1814"


def common_prefix(left: list[int], right: list[int]) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))
