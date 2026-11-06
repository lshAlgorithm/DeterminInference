from __future__ import annotations

import argparse
import concurrent.futures
import json
import urllib.error
import urllib.request
from pathlib import Path


def check_live_lifecycle(url: str, traces: Path) -> dict:
    def generate(prompt, length, salt, replay=None):
        params = {'temperature': 0, 'max_new_tokens': length, 'ignore_eos': True, 'sampling_seed': 42}
        if replay is not None:
            params['replay_token_ids'] = replay
        request = urllib.request.Request(url + '/generate', json.dumps({'input_ids': prompt, 'extra_key': salt, 'sampling_params': params}).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)

    trace = json.loads(sorted(traces.glob('*.json'))[0].read_text())
    first, branch, joined, reset = trace['requests']
    ordered = [first, *branch['requests'], joined, reset]
    reference = []
    for i, row in enumerate(ordered):
        result = generate(row['input_ids'], len(row['output_ids']), 'lifecycle-from-start', row['output_ids'])
        assert result['output_ids'] == row['output_ids']
        meta = result['meta_info']
        reference.append({'row': i, 'prompt': meta['prompt_tokens'], 'cached': meta['cached_tokens'], 'output': meta['completion_tokens']})
    assert reference[0]['cached'] == reference[-1]['cached'] == 0
    assert reference[1]['cached'] >= 207 and reference[2]['cached'] >= 230 and reference[3]['cached'] >= 207
    prompt = first['input_ids']
    ordinary = generate(prompt, 13, 'lifecycle-normal-reference')['output_ids']
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(generate, prompt, 3, 'lifecycle-mix-a', [7, 8, 9]), pool.submit(generate, prompt, 7, 'lifecycle-mix-b', [10]*7), pool.submit(generate, prompt, 13, 'lifecycle-mix-c')]
        mixed = [job.result() for job in jobs]
    assert mixed[0]['output_ids'] == [7, 8, 9] and mixed[1]['output_ids'] == [10]*7
    assert mixed[2]['output_ids'] == ordinary
    after = generate(prompt, 13, 'lifecycle-normal-after')
    assert after['output_ids'] == ordinary
    rejected = []
    for prompt_size, out_size in [(16383, 2), (16379, 1)]:
        try:
            result = generate([500] * prompt_size, out_size, 'lifecycle-budget', [7]*out_size)
        except urllib.error.HTTPError as error:
            body = error.read().decode()
            assert error.code == 400 and 'token replay' in body and ('budget' in body or 'truncation' in body), (error.code, body)
            rejected.append({'status': error.code, 'body': body})
        else:
            assert result['meta_info']['finish_reason']['type'] == 'abort', result
            rejected.append(result['meta_info']['finish_reason'])
    return {'passed': True, 'seed': 42, 'reference_from_start': reference, 'mixed_lengths': [len(x['output_ids']) for x in mixed], 'normal_after_mixed_matches': True, 'oversized_rejections': rejected, 'scope': 'sequential reference, mixed lifecycle and HTTP admission correctness; no performance claim'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True)
    parser.add_argument('--traces', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = check_live_lifecycle(args.url, args.traces)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
