from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def compare_profiles(summary: dict, specification: dict) -> dict:
    expected = {(Path(session).stem, row, rank) for session, rows in specification['profile_targets'].items() for row in rows for rank in [0, 1]}
    arms = {'real': {}, 'synthetic': {}}
    for profile in summary['profiles']:
        path = Path(profile['file'])
        content = path.parts[0].removeprefix('representative-')
        match = re.fullmatch(r'(\d+)-row(\d+)-[\d.]+-TP-(\d+)-EP-(\d+)\.trace\.json(?:\.gz)?', path.name)
        if content not in arms or match is None:
            raise ValueError(f'unexpected trace: {path}')
        session, row, rank, ep = match.groups()
        key = (session, int(row), int(rank))
        if rank != ep or key in arms[content]:
            raise ValueError('unexpected or duplicate rank trace')
        arms[content][key] = profile
    if any(set(arm) != expected for arm in arms.values()):
        raise ValueError('representative trace coverage incomplete')
    pairs = []
    totals = {content: {rank: {} for rank in [0, 1]} for content in arms}
    for key in sorted(expected):
        pair = {'session': key[0]+'.json', 'request': key[1], 'rank': key[2], 'arms': {}}
        for content, arm in arms.items():
            profile = arm[key]
            values = {'kernel_span_ms': profile['kernel_span_ms']}
            for group in ['communication', 'compute_and_other']:
                for kind in ['kernel_sum_ms', 'union_ms']:
                    values[group+'_'+kind] = profile['durations'].get(group, {}).get(kind, 0)
            pair['arms'][content] = values
            for metric, value in values.items():
                totals[content][key[2]][metric] = totals[content][key[2]].get(metric, 0)+value
        pairs.append(pair)
    return {'complete': True, 'representative_requests_per_content': len(expected)//2, 'paired_rank_traces': pairs,
            'selected_request_sums_by_rank': totals,
            'scope': 'nine prespecified full requests per content, two ranks; profiler-on diagnostics only; unions are within each category/request, not additive across categories or ranks; sums across selected requests are not full-workload GPU time; no repeated-profile uncertainty estimate'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(compare_profiles(json.loads(args.summary.read_text()), json.loads(args.spec.read_text())), indent=2))
