from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def aggregate_routing(routing: dict, sessions: int, ep_size: int = 2) -> dict:
    ranks = defaultdict(list)
    for record in routing['rank_records']:
        rank = int(Path(record['file']).stem.rsplit('_', 1)[1])
        ranks[rank].append(record)
    if set(ranks) != set(range(ep_size)):
        raise ValueError('missing or unexpected recorder ranks')
    totals = {}
    for rank, records in ranks.items():
        if len(records) != sessions:
            raise ValueError(f'rank {rank}: expected {sessions} session dumps, got {len(records)}')
        phases = {}
        for record in sorted(records, key=lambda r: float(Path(r['file']).stem.split('_')[-2])):
            for phase, data in record['phases'].items():
                hist = data['per_layer_physical_expert_counts']
                if not hist or not hist[0] or len(hist[0]) % ep_size:
                    raise ValueError('invalid expert dimensions')
                if phase not in phases:
                    phases[phase] = [[0 for _ in layer] for layer in hist]
                if [len(layer) for layer in hist] != [len(layer) for layer in phases[phase]]:
                    raise ValueError('inconsistent histogram dimensions')
                for dest, layer in zip(phases[phase], hist, strict=True):
                    for index, count in enumerate(layer):
                        if count < 0 or int(count) != count:
                            raise ValueError('invalid expert count')
                        dest[index] += count
        totals[rank] = phases
    if any(totals[rank] != totals[0] for rank in ranks):
        raise ValueError('replicated global rank histograms disagree')
    result = {}
    for phase, hist in totals[0].items():
        layers = []
        for counts in hist:
            width = len(counts) // ep_size
            loads = [sum(counts[rank*width:(rank+1)*width]) for rank in range(ep_size)]
            mean = statistics.mean(counts)
            card_mean = statistics.mean(loads)
            layers.append({'expert_counts': counts, 'expert_max_over_mean': max(counts)/mean if mean else None,
                           'expert_cv': statistics.pstdev(counts)/mean if mean else None,
                           'card_token_load': loads, 'card_max_over_mean': max(loads)/card_mean if card_mean else None,
                           'card_cv': statistics.pstdev(loads)/card_mean if card_mean else None})
        result[phase] = layers
    return {'sessions': sessions, 'replicas_equal': True, 'selected_rank': 0, 'phases': result,
            'scope': 'session sums from one verified global replica, fixed contiguous expert placement; routed token counts are not GPU execution time'}


def summarize_service(directory: Path, expected: dict) -> dict:
    summary = json.loads((directory/'summary.json').read_text())
    if summary['profile']:
        raise ValueError('profiled arm in performance comparison')
    rows = [json.loads(line) for line in (directory/'requests.jsonl').read_text().splitlines()]
    keys = [(r['session'], r['request']) for r in rows]
    if len(set(keys)) != len(rows) or set(keys) != set(expected):
        raise ValueError(f'incomplete or duplicated request coverage: {directory}')
    for row in rows:
        if row.get('profiled', summary['profile']) or not row['output_ids_exact']:
            raise ValueError('profiled or inexact request in performance comparison')
        if [row['prompt_tokens'], row['output_tokens']] != expected[(row['session'], row['request'])]:
            raise ValueError('request lengths disagree with frozen data')
        if not 0 <= row['cached_tokens'] <= row['prompt_tokens'] or row['wall_seconds'] <= 0:
            raise ValueError('invalid cache count or timing')
    work = {f'{r["session"]}:{r["request"]}': [r['prompt_tokens'], r['output_tokens'], r['cached_tokens']] for r in rows}
    seconds = sum(r['wall_seconds'] for r in rows)
    tokens = sum(r['output_tokens'] for r in rows)
    decode_tokens = sum(max(r['output_tokens']-1, 0) for r in rows)
    return {'requests': len(rows), 'output_tokens': tokens, 'prompt_tokens': sum(r['prompt_tokens'] for r in rows),
            'cached_tokens': sum(r['cached_tokens'] for r in rows), 'service_seconds': seconds,
            'service_output_tps': tokens/seconds, 'per_gpu_service_output_tps': tokens/seconds/2,
            'mean_ttft_seconds': statistics.mean(r['ttft_seconds'] for r in rows),
            'mean_request_seconds': seconds/len(rows),
            'token_weighted_tpot_seconds': sum(r['decode_seconds'] for r in rows)/decode_tokens,
            'work_by_request': work}


def compare_moe(off: Path, on: Path | None) -> dict:
    state = json.loads((off/'status.json').read_text())
    if state['status'] != 'complete':
        raise ValueError('recorder-off study incomplete')
    expected = {}
    for content, source in state['spec']['sources'].items():
        expected[content] = {(p.name, i): [len(r['input_ids']), len(r['output_ids'])]
                             for p in sorted(Path(source).glob('*.json'))
                             for i, r in enumerate(json.loads(p.read_text())['requests'])}
    if not expected['real'] or expected['real'] != expected['synthetic']:
        raise ValueError('source length structure differs or is empty')
    arms = {}
    pairs = []
    metrics = ['service_seconds', 'service_output_tps', 'mean_ttft_seconds', 'token_weighted_tpot_seconds']
    for repeat in range(3):
        for content in ['real', 'synthetic']:
            name = f'repeat{repeat}-{content}'
            arms[name] = summarize_service(off/name, expected[content])
        real, synthetic = [arms[f'repeat{repeat}-{c}'] for c in ['real', 'synthetic']]
        if real['work_by_request'] != synthetic['work_by_request']:
            raise ValueError('paired real/synthetic cached work differs')
        pairs.append({'repeat': repeat, 'synthetic_relative_to_real': {m: synthetic[m]/real[m]-1 for m in metrics}})
    estimates = {}
    for metric in metrics:
        values = [p['synthetic_relative_to_real'][metric] for p in pairs]
        mean = statistics.mean(values)
        half = 4.302652729911275*statistics.stdev(values)/math.sqrt(3)
        estimates[metric] = {'mean': mean, 'ci95': [mean-half, mean+half]}
    perturbation = {}
    routing = {}
    if on is not None:
        if json.loads((on/'status.json').read_text())['status'] != 'complete':
            raise ValueError('recorder-on study incomplete')
        for content in ['real', 'synthetic']:
            recorded = summarize_service(on/f'routing-{content}', expected[content])
            reference = [arms[f'repeat{i}-{content}'] for i in range(3)]
            if any(recorded['work_by_request'] != r['work_by_request'] for r in reference):
                raise ValueError('recorder on/off cached work differs')
            perturbation[content] = {'on': recorded, 'relative_to_off_mean': {m: recorded[m]/statistics.mean(r[m] for r in reference)-1 for m in metrics},
                                     'scope': 'one recorder-on pass versus three earlier off passes; descriptive perturbation, no paired uncertainty or drift control'}
            routing[content] = aggregate_routing(json.loads((on/f'routing-{content}'/'routing.json').read_text()), len({k[0] for k in expected[content]}))
        for phase in routing['real']['phases']:
            for real, synthetic in zip(routing['real']['phases'][phase], routing['synthetic']['phases'][phase], strict=True):
                a, b = real['expert_counts'], synthetic['expert_counts']
                real['synthetic_total_variation_distance'] = 0.5*sum(abs(x/sum(a)-y/sum(b)) for x, y in zip(a,b,strict=True)) if sum(a) and sum(b) else None
    return {'complete_off': True, 'complete_on': on is not None, 'arms': arms, 'pairs': pairs, 'estimates': estimates,
            'recorder_perturbation': perturbation, 'routing': routing,
            'scope': 'fixed zero-wait sequential service, cache flushed between sessions; three-pair t intervals for repeated runs, no population or source-arrival claims; excludes flush, initialization and file I/O; token routing differences do not establish GPU timing differences'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--off', type=Path, required=True)
    parser.add_argument('--on', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(compare_moe(args.off, args.on), indent=2))
