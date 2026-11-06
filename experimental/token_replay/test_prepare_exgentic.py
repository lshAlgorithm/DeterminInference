import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import prepare_exgentic as prepare


def make_span(text="answer", **attributes):
    return {
        "start_time": "2026-01-01T00:00:00+00:00",
        "end_time": "2026-01-01T00:00:01+00:00",
        "attributes": {
            "gen_ai.operation.name": "chat",
            "gen_ai.input.messages": json.dumps([{"role": "user", "parts": [{"type": "text", "content": "question"}]}]),
            "gen_ai.output.messages": json.dumps([{"role": "assistant", "parts": [{"type": "text", "content": text}]}]),
            **attributes,
        },
    }


def make_tokenizer():
    def render(messages, tokenize, add_generation_prompt, tools=None):
        prefix = json.dumps(tools) if tools else ""
        body = "".join("<" + message["role"] + ">" + message["content"] + "</>" for message in messages)
        return prefix + body + ("<assistant>" if add_generation_prompt else "")
    return SimpleNamespace(chat_template="test", apply_chat_template=render, encode=lambda text, **kwargs: list(map(ord, text)))
