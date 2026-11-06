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
