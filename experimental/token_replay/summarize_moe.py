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
