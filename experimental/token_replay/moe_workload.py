from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from moe_probe import guard_gpus, summarize_recorder


def run_workload(engine, source: Path, output: Path, record: bool, profile: bool) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for path in sorted(source.glob('*.json')):
        trace = json.loads(path.read_text())
        engine.flush_cache()
        if record:
            engine.start_expert_distribution_record()
        with (output / 'requests.jsonl').open('a') as stream:
            for index, request in enumerate(trace['requests']):
                if profile and (path.name != '000.json' or index != 0):
                    continue
                if profile:
                    engine.start_profile(output_dir=str(output), activities=['CPU', 'GPU'], with_stack=False, record_shapes=False, profile_prefix='rank-timing')
                params = {'temperature': 0, 'max_new_tokens': len(request['output_ids']), 'ignore_eos': True, 'sampling_seed': 42, 'replay_token_ids': request['output_ids']}
                started = time.perf_counter()
                first = None
                result = None
                for result in engine.generate(input_ids=request['input_ids'], sampling_params=params, stream=True):
                    if first is None and result.get('output_ids'):
                        first = time.perf_counter()
                ended = time.perf_counter()
                if profile:
                    engine.stop_profile()
                assert result['output_ids'] == request['output_ids'], (path.name, index, result)
                assert result['meta_info']['finish_reason']['type'] == 'length'
                meta = result['meta_info']
                row = {'session': path.name, 'request': index, 'prompt_tokens': len(request['input_ids']), 'output_tokens': len(request['output_ids']), 'cached_tokens': meta['cached_tokens'], 'wall_seconds': ended-started, 'ttft_seconds': first-started, 'decode_seconds': ended-first, 'output_ids_exact': True, 'meta_info': meta}
                stream.write(json.dumps(row)+'\n')
                stream.flush()
                rows.append(row)
        if record:
            engine.stop_expert_distribution_record()
            engine.dump_expert_distribution_record()
        print(json.dumps({'directory': str(output), 'session': path.name, 'completed_requests': len(rows)}), flush=True)
        if profile:
            break
    summary = {'source': str(source), 'requests': len(rows), 'output_tokens': sum(r['output_tokens'] for r in rows), 'prompt_tokens': sum(r['prompt_tokens'] for r in rows), 'cached_tokens': sum(r['cached_tokens'] for r in rows), 'request_wall_seconds': sum(r['wall_seconds'] for r in rows), 'record': record, 'profile': profile, 'scope': 'fixed ordered sequential windows; cache flushed between sessions; source timings are processed, no original DAG claims'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary
