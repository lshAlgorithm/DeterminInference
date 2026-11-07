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


def load_qualified_rows(archive: Path, manifest_path: Path, reference_path: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text())
    reference = json.loads(reference_path.read_text())
    if manifest.get("status") != "qualified" or manifest.get("selected_requests") != 447:
        raise ValueError("expected the qualified 447-request Dense source manifest")
    if not reference.get("complete") or reference.get("requests_exact") != 447:
        raise ValueError("the frozen corpus reference must cover all 447 requests")
    references = {(r["session"], r["request"]): r for r in reference["rows"]}
    if len(references) != 447:
        raise ValueError("duplicate or missing corpus reference identities")
    rows = []
    with tarfile.open(archive) as source:
        for selected in manifest["selected"]:
            member = "exgentic-dense-02/" + selected["file"]
            stream = source.extractfile(member)
            if stream is None:
                raise ValueError(f"missing source member: {member}")
            raw = stream.read(16 * 1024 * 1024 + 1)
            if len(raw) > 16 * 1024 * 1024:
                raise ValueError("source session exceeds bounded input size")
            if hashlib.sha256(raw).hexdigest() != selected["sha256"]:
                raise ValueError(f"source manifest hash mismatch: {member}")
            requests = json.loads(raw)["requests"]
            if len(requests) != len(selected["requests"]):
                raise ValueError("source request count differs from manifest")
            for index, request in enumerate(requests):
                identity = (Path(selected["file"]).name, index)
                expected = references.get(identity)
                if expected is None or request["output_ids"] != expected["output_ids"]:
                    raise ValueError(f"source/reference output mismatch: {identity}")
                if request.get("type") != "s" or request["in"] != len(request["input_ids"]) or request["out"] != len(request["output_ids"]):
                    raise ValueError(f"incomplete source request: {identity}")
                vectors = request["input_ids"] + request["output_ids"]
                if any(type(token) is not int or token < 0 for token in vectors):
                    raise ValueError("invalid source token IDs")
                rows.append({"source_file": member, "source_row": index,
                             "source_sha256": selected["sha256"],
                             "source_session_id": selected["session_id"],
                             "source_span": selected["requests"][index],
                             "request": request})
    if len(rows) != 447 or len({(r["source_file"], r["source_row"]) for r in rows}) != 447:
        raise ValueError("source denominator must be 447 distinct complete requests")
    return rows


def build_spec(rows: list[dict], identity: dict) -> dict:
    required = ("model_sha256", "serving_source_sha256", "observer_sha256", "wheel_sha256")
    for key in required:
        value = identity.get(key, "")
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"identity requires a SHA256: {key}")
    eligible = sorted((r for r in rows if len(r["request"]["output_ids"]) >= 64
                       and len(r["request"]["input_ids"]) + len(r["request"]["output_ids"]) <= 16384),
                      key=lambda r: (r["source_file"], r["source_row"]))
    random.Random(42).shuffle(eligible)
    selected, lengths = [], set()
    for row in eligible:
        length = len(row["request"]["output_ids"])
        if length in lengths:
            continue
        lengths.add(length)
        selected.append({**row, "expected_length": length,
                         "input_sha256": canonical_hash(row["request"]["input_ids"]),
                         "output_sha256": canonical_hash(row["request"]["output_ids"])})
        if len(selected) == 8:
            break
    if len(selected) != 8:
        raise ValueError("eight distinct eligible lengths are required before execution")
    spec = {"schema_version": 1, "seed": 42, "identity": identity,
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_requests": len(rows), "selected": selected,
            "model": "Qwen2.5-7B-Instruct-GPTQ-Int4",
            "server": {"context_length": 16384, "max_total_tokens": 131072,
                       "max_running_requests": 16, "cuda_graph_max_bs": 16,
                       "attention_backend": "triton", "random_seed": 42,
                       "disable_cuda_graph": True},
            "trigger": {"SGLANG_TEST_RETRACT": "true", "SGLANG_TEST_RETRACT_INTERVAL": "3",
                        "SGLANG_TEST_RETRACT_NO_PREFILL_BS": "2147483648"},
            "attempts": [4, 8], "attempt_seconds": 600, "allocation_seconds": 1800,
            "io_timeout_seconds": 10, "cleanup_seconds": 30,
            "sampling": {"temperature": 0, "ignore_eos": True, "sampling_seed": 42},
            "ordinary_source_index": min(range(4), key=lambda i: selected[i]["expected_length"]),
            "schedule": {"mix": "longest replay, then c-2 shorter replays and one ordinary; peak c",
                         "cancel": "remaining selected source row, after mixed requests drain",
                         "followup": "ordinary and same-source fresh replay after terminal abort and retirement"},
            "scope": "controlled lifecycle correctness only; no timing or KV-equality claim"}
    spec["spec_sha256"] = canonical_hash(spec)
    return spec


