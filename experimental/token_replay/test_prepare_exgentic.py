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
            "gen_ai.input.messages": json.dumps(
                [{"role": "user", "parts": [{"type": "text", "content": "question"}]}]
            ),
            "gen_ai.output.messages": json.dumps(
                [{"role": "assistant", "parts": [{"type": "text", "content": text}]}]
            ),
            **attributes,
        },
    }


def make_tokenizer():
    def render(messages, tokenize, add_generation_prompt, tools=None):
        prefix = json.dumps(tools) if tools else ""
        body = "".join(
            "<" + message["role"] + ">" + message["content"] + "</>"
            for message in messages
        )
        return prefix + body + ("<assistant>" if add_generation_prompt else "")

    return SimpleNamespace(
        chat_template="test",
        apply_chat_template=render,
        encode=lambda text, **kwargs: list(map(ord, text)),
    )


class TestPreparation(unittest.TestCase):
    def test_invalid_call_breaks_continuous_window(self):
        requests = [{"start": index * 2, "end": index * 2 + 1} for index in range(7)]
        requests[3] = ValueError("rejected_call")
        with patch.object(prepare, "encode_span", side_effect=requests):
            window, receipt = prepare.qualify_session(
                {"session_id": "a", "spans": [make_span()] * 7}, None
            )
        self.assertEqual(window, [])
        self.assertEqual(
            receipt["exclusions"], [{"span_index": 3, "reason": "rejected_call"}]
        )
        with patch.object(
            prepare,
            "encode_span",
            side_effect=[
                {"start": index * 2, "end": index * 2 + 1} for index in range(4)
            ],
        ):
            window, receipt = prepare.qualify_session(
                {"session_id": "a", "spans": [make_span()] * 4}, None
            )
        self.assertEqual(receipt["selected_span_indexes"], [0, 1, 2, 3])

    def test_tool_calls_not_preserved_by_template_are_rejected(self):
        output = [
            {
                "role": "assistant",
                "parts": [
                    {
                        "type": "tool_call",
                        "name": "run",
                        "id": "c1",
                        "arguments": {"command": "ls"},
                    }
                ],
            }
        ]
        span = make_span(**{"gen_ai.output.messages": json.dumps(output)})
        with self.assertRaisesRegex(ValueError, "target_template_drops_tool_calls"):
            prepare.encode_span(span, make_tokenizer())

    def test_joint_encoding_and_bounds(self):
        result = prepare.encode_span(make_span(), make_tokenizer())
        self.assertEqual(result["output_ids"], list(map(ord, "answer</>")))
        for text, reason in [
            ("a" * 2049, "target_output_over_2048"),
            ("", "empty_source_response"),
        ]:
            with self.assertRaisesRegex(ValueError, reason):
                prepare.encode_span(make_span(text), make_tokenizer())
