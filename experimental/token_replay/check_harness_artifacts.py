import json
from pathlib import Path


def check_harness_artifacts(folder: Path) -> dict:
    source = [json.loads(line) for line in (folder / 'recorded-trace.jsonl').read_text().splitlines()]
    raw = [json.loads(line) for line in (folder / 'harness-real-02-raw.jsonl').read_text().splitlines()]
    raw.sort(key=lambda row: row['metadata']['turn_index'])
    outputs = json.loads((folder / 'harness-real-02-outputs.json').read_text())['data']
    metrics = json.loads((folder / 'harness-real-02-metrics.json').read_text())
    assert len(raw) == len(source) == len(outputs) == 2
    assert metrics['request_count']['avg'] == 2
    assert metrics['request_error_rate']['avg'] == 0
    assert metrics['error_summary'] == [] and metrics['was_cancelled'] is False
    for index, (turn, response, output) in enumerate(zip(source, raw, outputs)):
        assert response['status'] == 200
        assert response['metadata']['turn_index'] == index
        assert response['payload']['prompt'] == turn['input_ids']
        assert response['payload']['replay_token_ids'] == turn['output_ids']
        assert response['payload']['seed'] == 42
        chunks = []
        for event in response['responses']:
            for packet in event['packets']:
                if packet['name'] == 'data' and packet['value'] != '[DONE]':
                    chunks.append(json.loads(packet['value']))
        text = ''.join(choice['text'] for chunk in chunks for choice in chunk['choices'])
        assert text == output['response_text']
        usage = [chunk['usage'] for chunk in chunks if chunk.get('usage')]
        assert usage[-1]['completion_tokens'] == len(turn['output_ids'])
    assert raw[1]['metadata']['request_start_ns'] >= raw[0]['metadata']['request_end_ns']
    return {'requests': 2, 'http_statuses': [200, 200], 'exact_payload_ids': 'passed', 'stream_export_parity': 'passed', 'sequential_turns': 'passed', 'request_error_rate': 0}


if __name__ == '__main__':
    print(json.dumps(check_harness_artifacts(Path(__file__).parent / 'evidence'), indent=2))
