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
    rows = [json.loads(line) for line in (directory/'requests.jsonl').read_text().splitlines()]
    keys = [(r['session'], r['request']) for r in rows]
    if len(set(keys)) != len(rows) or set(keys) != set(expected):
        raise ValueError(f'incomplete or duplicated request coverage: {directory}')
    for row in rows:
        if row['profiled'] or not row['output_ids_exact']:
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
