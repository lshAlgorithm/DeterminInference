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

