from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_natural_spec(base_path: Path, output_path: Path) -> dict:
    base = json.loads(base_path.read_text())
    base_digest = base.pop("spec_sha256", None)
    if base_digest is None or canonical_hash(base) != base_digest:
        raise ValueError("base lifecycle specification hash mismatch")
    if base.get("seed") != 42 or base.get("attempts") != [4, 8]:
        raise ValueError("natural L1 must retain the frozen seed and two attempts")
    trigger = {
        "kind": "natural_kv_pressure",
        "retract_condition": "not batch.check_decode_mem()",
        "test_retract_environment": "unset",
        "pause_endpoint": "unused",
    }
    natural = {
        **base,
        "run_id": "natural-kv-l1-20261109",
        "trigger": trigger,
        "scope": "natural KV-pressure retract/resume correctness only; no performance claim",
        "output_root": "/home/tegao/infra-token-replay/natural-kv-l1-20261109",
        "launch_contract": {
            "cuda_visible_devices": "0",
            "sglang_test_retract": "unset",
            "sglang_test_retract_interval": "unset",
            "sglang_test_retract_no_prefill_bs": "unset",
            "server_port": 19080,
            "unique_request_namespace": "natural-b1-",
        },
    }
    natural["spec_sha256"] = canonical_hash(natural)
    output_path.write_text(json.dumps(natural, indent=2) + "\n")
    return natural


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = build_natural_spec(args.base, args.output)
    print(
        json.dumps(
            {
                "run_id": spec["run_id"],
                "spec_sha256": spec["spec_sha256"],
                "seed": spec["seed"],
                "selected": [
                    {
                        "file": row["source_file"],
                        "row": row["source_row"],
                        "length": row["expected_length"],
                    }
                    for row in spec["selected"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
