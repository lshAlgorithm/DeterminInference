from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

DATASET = "Exgentic/agent-llm-traces-v2"
REVISION = "4b8ad4ab198438e5a170f9171c19c6a2cf7c1814"
SOURCE_HASHES = [
    "f7cd079798f47b7610963c08105055b935316c6223f74d3fc03c401348590011",
    "ef6fb9129aae06a155cc01f1868a707f1544fe9ae11df3c04fff3e71a964ac2c",
    "d380ea81f8ac2ee05c0ace3a7ac7896272fc437e1e3b3a47d90de544b525477e",
    "e7ab71e7eae7dd44197a448dbee512407a2a0ae0382b58f5f8a36633edba7d9e",
    "283f16603e52d9d1c82b1a4a826f3ee7c09916bae036e950ed1bae683037fc80",
    "f17f9f64139860b628cab938771f91dd141952f14eb21f88fa62df335c0f4d84",
    "fd5e280a14b4b2116c1cdcb0b1cb945da66280d999f3c56f8848a74fdb6247e3",
    "d22806edf2baff6ce13a63b1f6347d7b30012624222344ebe8ed82f0d84ed3d6",
    "2aa8507830af585ac977820210adc4fc80abb7a0bbcb89bc23b6a780ff55d064",
]


def common_prefix(left: list[int], right: list[int]) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def prefix_hashes(input_ids: list[int], block_size: int = 16) -> list[int]:
    digest = hashlib.sha256()
    hashes = []
    for start in range(0, len(input_ids) - block_size + 1, block_size):
        digest.update(json.dumps(input_ids[start : start + block_size], separators=(",", ":")).encode())
        hashes.append(int.from_bytes(digest.digest()[:8], "big") >> 1)
    return hashes


def encode_span(span: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    from aiperf.dataset.loader.exgentic import (
        _normalize_messages,
        _normalize_system_instructions,
        _normalize_tools,
        _timestamp_ms,
    )

    attributes = span.get("attributes") or {}
    if (span.get("status") or {}).get("code") == 2:
        raise ValueError("failed_source_call")
    messages = _normalize_system_instructions(attributes.get("gen_ai.system_instructions"))
    messages += _normalize_messages(attributes.get("gen_ai.input.messages"))
    outputs = _normalize_messages(attributes.get("gen_ai.output.messages"))
    if len(outputs) != 1 or outputs[0]["role"] != "assistant":
        raise ValueError("output_is_not_one_assistant_message")
    if not outputs[0].get("content") and not outputs[0].get("tool_calls"):
        raise ValueError("empty_source_response")
    if any(message.get("reasoning_content") for message in messages + outputs):
        raise ValueError("reasoning_semantics_not_qualified_for_target_template")
    tools = _normalize_tools(attributes.get("gen_ai.tool.definitions"))
    template_args = {"tools": tools} if tools else {}
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **template_args
    )
    complete = tokenizer.apply_chat_template(
        messages + outputs, tokenize=False, add_generation_prompt=False, **template_args
    )
    if tools and prompt == tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True):
        raise ValueError("target_template_drops_tool_definitions")
    if any(message.get("tool_calls") for message in messages + outputs):
        without_calls = [{key: value for key, value in message.items() if key != "tool_calls"} for message in messages + outputs]
        if complete == tokenizer.apply_chat_template(without_calls, tokenize=False, add_generation_prompt=False, **template_args):
            raise ValueError("target_template_drops_tool_calls")
    for message in messages + outputs:
        if message.get("content") and message["content"] not in complete:
            raise ValueError("target_template_drops_message_content")
        for call in message.get("tool_calls", []):
            if call["function"]["name"] not in complete:
                raise ValueError("target_template_drops_tool_calls")
    for tool in tools or []:
        if tool["function"]["name"] not in prompt:
            raise ValueError("target_template_drops_tool_definitions")
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    complete_ids = tokenizer.encode(complete, add_special_tokens=False)
    if complete_ids[: len(input_ids)] != input_ids:
        raise ValueError("joint_encoding_prefix_mismatch")
    output_ids = complete_ids[len(input_ids) :]
    if not input_ids or not output_ids:
        raise ValueError("empty_target_token_sequence")
    if len(output_ids) > 2048:
        raise ValueError("target_output_over_2048")
    if len(complete_ids) > 16384:
        raise ValueError("target_context_over_16384")
    start = _timestamp_ms(span["start_time"]) / 1000
    end = _timestamp_ms(span["end_time"]) / 1000
    if not math.isfinite(start + end) or end < start:
        raise ValueError("invalid_source_timing")
    return {
        "start": start,
        "end": end,
        "input_ids": input_ids,
        "output_ids": output_ids,
        "source_model": attributes.get("gen_ai.request.model"),
    }


def qualify_session(row: dict[str, Any], tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from aiperf.dataset.loader.exgentic import _is_replayable_span

    best: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    exclusions = []
    llm_spans = 0
    spans = row.get("spans") or []
    visited = 0
    for span_index, span in enumerate(spans):
        visited += 1
        if not _is_replayable_span(span, span.get("attributes") or {}):
            continue
        llm_spans += 1
        try:
            request = encode_span(span, tokenizer)
            if current and request["start"] < current[-1]["end"]:
                raise ValueError("overlap_without_qualified_dag")
            request["source_span_index"] = span_index
            current.append(request)
            if len(current) > len(best):
                best = list(current)
            if len(best) == 16:
                break
        except (ValueError, KeyError, TypeError) as error:
            exclusions.append({"span_index": span_index, "reason": str(error)})
            current = []
    valid = best if len(best) >= 4 else []
    return valid, {
        "session_id": row.get("session_id"),
        "source_spans": len(spans),
        "visited_spans": visited,
        "unexamined_spans": len(spans) - visited,
        "examined_llm_spans": llm_spans,
        "exclusions": exclusions,
        "selected_span_indexes": [request["source_span_index"] for request in valid],
        "reason": "selected" if valid else "no_continuous_valid_window_of_four",
    }


def make_weka(session_id: str, requests: list[dict[str, Any]], model: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    origin = requests[0]["start"]
    previous: dict[str, Any] | None = None
    records = []
    provenance = []
    for request in requests:
        input_ids, output_ids = request["input_ids"], request["output_ids"]
        records.append({
            "type": "s",
            "t": request["start"] - origin,
            "model": model,
            "in": len(input_ids),
            "out": len(output_ids),
            "hash_ids": [
                int.from_bytes(hashlib.sha256(bytes(str(input_ids[:end]), "ascii")).digest()[:8], "big") >> 1
                for end in range(16, len(input_ids) + 1, 16)
            ],
            "input_ids": input_ids,
            "output_ids": output_ids,
            "api_time": request["end"] - request["start"],
            "think_time": max(0.0, request["start"] - previous["end"]) if previous else 0.0,
        })
        provenance.append({
            "source_span_index": request["source_span_index"],
            "source_model": request["source_model"],
            "input_tokens": len(input_ids),
            "output_tokens": len(output_ids),
            "previous_prompt_lcp": common_prefix(previous["input_ids"], input_ids) if previous else 0,
            "previous_prompt_response_lcp": common_prefix(previous["input_ids"] + previous["output_ids"], input_ids) if previous else 0,
        })
        previous = request
    return {
        "id": session_id,
        "models": [model],
        "block_size": 16,
        "hash_id_scope": "local",
        "requests": records,
    }, provenance


def source_candidates(folder: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import pyarrow.parquet as parquet

    sources = []
    candidates = []
    global_row = 0
    for shard_index, expected in enumerate(SOURCE_HASHES):
        path = folder / f"{shard_index:04d}.parquet"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"pinned source hash mismatch: {path}")
        source = parquet.ParquetFile(path)
        sources.append({"file": path.name, "sha256": digest, "bytes": path.stat().st_size, "rows": source.metadata.num_rows})
        shard_row = 0
        for group_index in range(source.num_row_groups):
            metadata = source.read_row_group(group_index, columns=["benchmark", "session_id"]).to_pylist()
            for group_row, row in enumerate(metadata):
                if row["benchmark"] == "swebench":
                    candidates.append({
                        "shard": path.name,
                        "row_group": group_index,
                        "group_row": group_row,
                        "shard_row": shard_row,
                        "global_row": global_row,
                        "session_id": row["session_id"],
                    })
                shard_row += 1
                global_row += 1
    return candidates, sources


def prepare(source_folder: Path, output: Path, tokenizer: Any, model: str, *, max_sessions: int = 32, max_examined_sessions: int = 128, seed: int = 42) -> dict[str, Any]:
    import pyarrow.parquet as parquet

    if output.exists():
        raise ValueError(f"output already exists: {output}")
    candidates, sources = source_candidates(source_folder)
    random.Random(seed).shuffle(candidates)
    output.mkdir(parents=True)
    session_dir = output / "sessions"
    session_dir.mkdir()
    manifest: dict[str, Any] = {
        "dataset": DATASET,
        "revision": REVISION,
        "benchmark": "swebench",
        "seed": seed,
        "target_model": model,
        "submission_valid": False,
        "token_ids": "retokenized_with_target_template_not_source_model_ids",
        "timing": "relative_to_selected_window; upstream_retry_adjusted; not_original_wall_clock",
        "dag": "not_recovered; sequential_nonoverlapping_calls_only",
        "selection": "seeded_session_order; longest_contiguous_valid_window_until_16; stop_at_session_budget",
        "limits": {"sessions": max_sessions, "examined_sessions": max_examined_sessions, "min_turns": 4, "max_turns": 16, "context_tokens": 16384, "output_tokens": 2048},
        "source_files": sources,
        "source_rows": sum(source["rows"] for source in sources),
        "candidate_swebench_sessions": len(candidates),
        "examined": [],
        "selected": [],
        "chat_template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
    }
    for candidate in candidates[:max_examined_sessions]:
        source = parquet.ParquetFile(source_folder / candidate["shard"])
        row = source.read_row_group(candidate["row_group"]).to_pylist()[candidate["group_row"]]
        requests, qualification = qualify_session(row, tokenizer)
        manifest["examined"].append({**candidate, **qualification})
        if requests:
            session_id = row["session_id"]
            trace, provenance = make_weka(session_id, requests, model)
            filename = f"{len(manifest['selected']):03d}.json"
            payload = json.dumps(trace, separators=(",", ":")).encode() + b"\n"
            (session_dir / filename).write_bytes(payload)
            manifest["selected"].append({**candidate, "file": f"sessions/{filename}", "sha256": hashlib.sha256(payload).hexdigest(), "requests": provenance})
        print(json.dumps({"examined": len(manifest["examined"]), "selected": len(manifest["selected"]), "session_id": row["session_id"], "reason": qualification["reason"]}), flush=True)
        if len(manifest["selected"]) >= max_sessions:
            break
    manifest["unexamined_candidate_sessions"] = len(candidates) - len(manifest["examined"])
    manifest["selected_sessions"] = len(manifest["selected"])
    manifest["selected_requests"] = sum(len(session["requests"]) for session in manifest["selected"])
    manifest["exclusion_counts"] = dict(Counter(exclusion["reason"] for row in manifest["examined"] for exclusion in row["exclusions"]))
    manifest["status"] = "qualified" if manifest["selected"] else "no_qualified_windows"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-folder", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-sessions", type=int, default=32)
    parser.add_argument("--max-examined-sessions", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.max_sessions <= 32 or args.max_examined_sessions < args.max_sessions:
        parser.error("require 1 <= max-sessions <= 32 and examined budget >= max-sessions")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    manifest = prepare(args.source_folder, args.output, tokenizer, args.model, max_sessions=args.max_sessions, max_examined_sessions=args.max_examined_sessions, seed=args.seed)
    print(json.dumps({key: manifest[key] for key in ("status", "selected_sessions", "selected_requests", "exclusion_counts")}, indent=2))


if __name__ == "__main__":
    main()
