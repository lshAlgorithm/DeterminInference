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
    if not __import__("math").isfinite(start + end) or end < start:
        raise ValueError("invalid_source_timing")
    return {
        "start": start,
        "end": end,
        "input_ids": input_ids,
        "output_ids": output_ids,
        "source_model": attributes.get("gen_ai.request.model"),
    }
