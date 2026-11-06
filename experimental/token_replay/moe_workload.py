from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from moe_probe import guard_gpus, summarize_recorder


def run_workload(engine, source: Path, output: Path, record: bool, profile: bool, profile_targets: dict | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    targets = profile_targets if profile_targets is not None else {'000.json': [0]}
    for path in sorted(source.glob('*.json')):
        if profile and path.name not in targets:
            continue
        trace = json.loads(path.read_text())
        engine.flush_cache()
        if record:
            engine.start_expert_distribution_record()
        with (output / 'requests.jsonl').open('a') as stream:
            for index, request in enumerate(trace['requests']):
                if profile and index > max(targets[path.name]):
                    break
                profiling = profile and index in targets[path.name]
                profile_stopped = False
                if profiling:
                    engine.start_profile(output_dir=str(output), activities=['CPU', 'GPU'], with_stack=False, record_shapes=False, profile_prefix=f'{path.stem}-row{index}')
                params = {'temperature': 0, 'max_new_tokens': len(request['output_ids']), 'ignore_eos': True, 'sampling_seed': 42, 'replay_token_ids': request['output_ids']}
                started = time.perf_counter()
                first = None
                result = None
                for result in engine.generate(input_ids=request['input_ids'], sampling_params=params, stream=True):
                    if first is None and result.get('output_ids'):
                        first = time.perf_counter()
                    if profiling and not profile_stopped and len(result.get('output_ids', [])) >= 32:
                        engine.stop_profile()
                        profile_stopped = True
                ended = time.perf_counter()
                if profiling and not profile_stopped:
                    engine.stop_profile()
                assert result['output_ids'] == request['output_ids'], (path.name, index, result)
                assert result['meta_info']['finish_reason']['type'] == 'length'
                meta = result['meta_info']
                row = {'profiled': profiling, 'session': path.name, 'request': index, 'prompt_tokens': len(request['input_ids']), 'output_tokens': len(request['output_ids']), 'cached_tokens': meta['cached_tokens'], 'wall_seconds': ended-started, 'ttft_seconds': first-started, 'decode_seconds': ended-first, 'output_ids_exact': True, 'meta_info': meta}
                stream.write(json.dumps(row)+'\n')
                stream.flush()
                rows.append(row)
        if record:
            engine.stop_expert_distribution_record()
            engine.dump_expert_distribution_record()
        print(json.dumps({'directory': str(output), 'session': path.name, 'completed_requests': len(rows)}), flush=True)
    summary = {'source': str(source), 'requests': len(rows), 'output_tokens': sum(r['output_tokens'] for r in rows), 'prompt_tokens': sum(r['prompt_tokens'] for r in rows), 'cached_tokens': sum(r['cached_tokens'] for r in rows), 'request_wall_seconds': sum(r['wall_seconds'] for r in rows), 'record': record, 'profile': profile, 'profiled_requests': sum(r['profiled'] for r in rows), 'profile_targets': targets if profile else None, 'scope': 'fixed ordered zero-wait sequential service measurement; source think times omitted; cache flushed between sessions; no original arrival-time or DAG claims'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR'] = str(args.output)
    import sglang as sgl
    status = {'status': 'starting', 'seed': 42, 'gpu_guard': guard_gpus(), 'spec': spec, 'completed': []}
    engine = None
    try:
        started = time.perf_counter()
        engine = sgl.Engine(**spec['settings'])
        status['initialization_seconds'] = time.perf_counter()-started
        status['server_info'] = engine.get_server_info()
        for source in spec['sources'].values():
            request = json.loads(sorted(Path(source).glob('*.json'))[0].read_text())['requests'][0]
            engine.generate(input_ids=request['input_ids'], sampling_params={'temperature':0, 'max_new_tokens':len(request['output_ids']), 'ignore_eos':True,'sampling_seed':42,'replay_token_ids':request['output_ids']})
            engine.flush_cache()
        status['status'] = 'running'
        for run in spec['runs']:
            status['current'] = run['name']
            (args.output/'status.json').write_text(json.dumps(status,indent=2))
            before = set(args.output.glob('expert_distribution_recorder_*.pt'))
            summary = run_workload(engine, Path(spec['sources'][run['content']]), args.output/run['name'], run['record'], run['profile'], spec.get('profile_targets'))
            if run['record']:
                for path in set(args.output.glob('expert_distribution_recorder_*.pt')) - before:
                    path.rename(args.output/run['name']/path.name)
                routing = summarize_recorder(args.output/run['name'], 2)
                (args.output/run['name']/'routing.json').write_text(json.dumps(routing,indent=2))
            status['completed'].append(summary)
        status['status'] = 'complete'
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        raise
    finally:
        if engine is not None:
            engine.shutdown()
        (args.output/'status.json').write_text(json.dumps(status,indent=2))


if __name__ == '__main__':
    main()
