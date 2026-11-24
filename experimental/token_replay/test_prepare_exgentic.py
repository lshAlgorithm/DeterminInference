import hashlib
import json
import tempfile
import unittest
from pathlib import Path
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
    def test_frozen_export_is_reproducible_and_rejects_source_changes(self):
        import pyarrow
        from pyarrow import parquet

        spans = [make_span() for _ in range(4)]
        for index, span in enumerate(spans):
            span["start_time"] = f"2026-01-01T00:00:{index * 2:02d}+00:00"
            span["end_time"] = f"2026-01-01T00:00:{index * 2 + 1:02d}+00:00"
        rows = [
            {"session_id": str(index), "benchmark": "swebench", "spans": spans}
            for index in range(3)
        ]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "0000.parquet"
            parquet.write_table(
                pyarrow.Table.from_pylist(rows), source, row_group_size=1
            )
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            with patch.object(prepare, "SOURCE_HASHES", [digest]):
                first = prepare.prepare(
                    root, root / "first", make_tokenizer(), "target", max_sessions=2
                )
                second = prepare.prepare(
                    root, root / "second", make_tokenizer(), "target", max_sessions=2
                )
                self.assertEqual(first, second)
                self.assertEqual(first["selected_requests"], 8)
                self.assertEqual(first["unexamined_candidate_sessions"], 1)
                for entry in first["selected"]:
                    self.assertEqual(
                        (root / "first" / entry["file"]).read_bytes(),
                        (root / "second" / entry["file"]).read_bytes(),
                    )
                source.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "pinned source hash mismatch"):
                    prepare.source_candidates(root)

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
