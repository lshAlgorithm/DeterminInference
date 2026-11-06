from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

from prepare_exgentic import prefix_hashes


def build_acceptance_trace(source: Path, output: Path) -> dict:
    """Derive small, token-consistent DAG fixtures from the bundled Weka example."""
    original = json.loads(source.read_text())
    trace = copy.deepcopy(original)
    rng = random.Random(42)
    tokens = [rng.randrange(1000, 50000) for _ in range(1024)]
    first, branch, joined = trace["requests"]
    child = branch["requests"][0]
    child_next = copy.deepcopy(child)
    reset = copy.deepcopy(joined)
    branch["requests"].append(child_next)
    trace["requests"].append(reset)
    first_ids = tokens[:192]
    first_out = tokens[192:208]
    child_ids = tokens[:224]
    child_out = tokens[224:231]
    child_next_ids = tokens[:256]
    joined_ids = tokens[:208] + tokens[320:368]
    rows = [
        (first, 0.0, first_ids, first_out),
        (child, 2.0, child_ids, child_out),
        (child_next, 3.0, child_next_ids, tokens[256:267]),
        (joined, 6.0, joined_ids, tokens[368:392]),
        (reset, 9.0, tokens[512:608], tokens[608:611]),
    ]
    for row, timestamp, inputs, outputs in rows:
        row.update(
            t=timestamp, model="Qwen2.5-7B-Instruct-GPTQ-Int4",
            input_ids=inputs, output_ids=outputs, hash_ids=prefix_hashes(inputs),
            api_time=0.25, think_time=0.5,
        )
        row["in"], row["out"] = len(inputs), len(outputs)
    branch["duration_ms"] = 2000
    branch["models"] = ["Qwen2.5-7B-Instruct-GPTQ-Int4"]
    trace["models"] = branch["models"]
    trace["block_size"] = 16
    output.mkdir(parents=True, exist_ok=False)
    sessions = output / "sessions"
    sessions.mkdir()
    for index in range(2):
        trace["id"] = f"derived-dag-{index}"
        (sessions / f"{index}.json").write_text(json.dumps(trace) + "\n")
    receipt = {
        "seed": 42, "source": str(source), "sessions": 2,
        "requests_per_session": 5, "submission_valid": False,
        "content": "synthetic token-consistent fixture; not original Weka content",
        "derivation": "bundled one_subagent topology with a second child turn and root reset",
        "timing": "fixture times; not measured workload timing",
        "expected_parent_child_prefix_tokens": 208,
        "expected_root_continuation_prefix_tokens": 208,
        "expected_child_continuation_prefix_tokens": 231,
        "independent_sessions_have_identical_ids": True,
    }
    (output / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_acceptance_trace(args.source, args.output), indent=2))
