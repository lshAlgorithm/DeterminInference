import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from infx.bench.env import InputError
from infx.bench.token_replay import load_sessions, main, replay_session


@pytest.fixture
def serving_endpoint():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(payload)
            ids = [8, 9] if payload["input_ids"] == [1, 2] else [7]
            body = json.dumps({"output_ids": ids, "meta_info": {"cached_tokens": 0}}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", received
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_replay_client_sends_full_tokens_and_keeps_turn_order(tmp_path, serving_endpoint):
    url, received = serving_endpoint
    rows = [
        {"session_id": "a", "input_ids": [1, 2], "output_ids": [8, 9], "delay_seconds": 0},
        {"session_id": "a", "input_ids": [1, 2, 8, 9, 3], "output_ids": [7], "delay_seconds": 0},
    ]
    source, output = tmp_path / "trace.jsonl", tmp_path / "result.json"
    source.write_text("\n".join(json.dumps(row) for row in rows))
    assert (
        main(
            [
                "--url",
                url,
                "--trace",
                str(source),
                "--output",
                str(output),
                "--concurrency",
                "2",
                "--timeout",
                "10",
                "--seed",
                "42",
            ]
        )
        == 0
    )
    assert received == [
        {
            "input_ids": row["input_ids"],
            "sampling_params": {
                "temperature": 0,
                "ignore_eos": True,
                "sampling_seed": 42,
                "max_new_tokens": len(row["output_ids"]),
                "replay_token_ids": row["output_ids"],
            },
        }
        for row in rows
    ]
    report = json.loads(output.read_text())
    assert [r["output_tokens"] for r in report["results"]] == [2, 1]
    assert report["unofficial"] is True


@pytest.mark.parametrize(
    "change",
    [
        {"hash_ids": [1]},
        {"output_ids": []},
        {"input_ids": [True]},
        {"delay_seconds": -1},
        {"delay_seconds": float("nan")},
    ],
)
def test_replay_loader_rejects_non_token_or_invalid_traces(tmp_path, change):
    row = {"session_id": "a", "input_ids": [1], "output_ids": [2], "delay_seconds": 0}
    row.update(change)
    source = tmp_path / "trace.jsonl"
    source.write_text(json.dumps(row))
    with pytest.raises(InputError):
        load_sessions(source)


def test_replay_stops_on_server_output_drift(serving_endpoint):
    url, received = serving_endpoint
    turns = [{"session_id": "a", "input_ids": [1, 2], "output_ids": [0], "delay_seconds": 0}]
    with pytest.raises(InputError, match="output IDs do not match"):
        replay_session(url, turns * 2, 10, 42)
    assert len(received) == 1
