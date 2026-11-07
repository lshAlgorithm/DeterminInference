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


