from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from prepare_exgentic import common_prefix, prefix_hashes


def prepare_synthetic(source: Path, output: Path, model: Path) -> dict:
    """Sample repository corpus tokens while preserving every within-session prefix."""
    from transformers import AutoTokenizer
    from aiperf.common.hash_id_random_generator import HashIdRandomGenerator
    from aiperf.dataset.generator import prompt as prompt_module

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    corpus_path = Path(prompt_module.__file__).parent / "assets/shakespeare.txt"
    special = set(tokenizer.all_special_ids)
    corpus = [
        token
        for token in tokenizer.encode(corpus_path.read_text(), add_special_tokens=False)
        if token not in special
    ]
    output.mkdir(parents=True, exist_ok=False)
    sessions = output / "sessions"
    sessions.mkdir()
    receipt = {
        "seed": 42,
        "source": str(source),
        "corpus": str(corpus_path),
        "method": "repository corpus sampler with a prefix trie and sibling collision rejection",
        "preserved": [
            "input/output token lengths",
            "all within-session pairwise LCPs",
            "special token positions",
            "timestamps and topology",
        ],
        "submission_valid": False,
        "sessions": [],
    }
    for path in sorted(source.glob("*.json")):
        trace = json.loads(path.read_text())
        rng = HashIdRandomGenerator(42, _internal=True)
        rng.set_trace_id(trace["id"])
        rng.reseed_for_hash_id(0)
        trie = {}
        originals, synthetic = [], []
        for row in trace["requests"]:
            if row["type"] == "subagent":
                raise ValueError(
                    "this paired Exgentic control expects qualified sequential windows"
                )
            old = row["input_ids"] + row["output_ids"]
            node = trie
            new = []
            for index, token in enumerate(old):
                if token in node:
                    replacement, node = node[token]
                    new.append(replacement)
                    continue
                used = {item[0] for item in node.values()}
                sampled = prompt_module.sample_tokens_from_corpus(
                    corpus, len(old) - index, rng
                )
                while token not in special and sampled[0] in used:
                    sampled = prompt_module.sample_tokens_from_corpus(
                        corpus, len(old) - index, rng
                    )
                for original, replacement in zip(old[index:], sampled, strict=True):
                    replacement = original if original in special else replacement
                    child = {}
                    node[original] = (replacement, child)
                    node = child
                    new.append(replacement)
                break
            split = len(row["input_ids"])
            originals.extend([old[:split], old])
            synthetic.extend([new[:split], new])
            row["input_ids"], row["output_ids"] = new[:split], new[split:]
            row["hash_ids"] = prefix_hashes(row["input_ids"])
        checked = 0
        for left, right in itertools.combinations(range(len(originals)), 2):
            assert common_prefix(originals[left], originals[right]) == common_prefix(
                synthetic[left], synthetic[right]
            )
            checked += 1
        assert [len(seq) for seq in originals] == [len(seq) for seq in synthetic]
        (sessions / path.name).write_text(
            json.dumps(trace, separators=(",", ":")) + "\n"
        )
        receipt["sessions"].append(
            {
                "file": path.name,
                "requests": len(trace["requests"]),
                "pairwise_prefix_checks": checked,
            }
        )
        print(json.dumps(receipt["sessions"][-1]), flush=True)
    receipt["matched"] = True
    (output / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    prepare_synthetic(args.source, args.output, args.model)
