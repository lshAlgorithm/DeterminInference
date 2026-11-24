from __future__ import annotations

import argparse
import json
from pathlib import Path


def audit_usage(summary: dict, source: Path) -> dict:
    expected = {}
    for path in sorted(source.glob('*.json')):
        trace = json.loads(path.read_text())
        for index, row in enumerate(trace['requests']):
            key = (trace['id'], index)
            if key in expected:
                raise ValueError('duplicate source request identity')
            expected[key] = (len(row['input_ids']), len(row['output_ids']))
    if not expected or not summary['points']:
        raise ValueError('empty source or campaign summary')
    checked = []
    for point in summary['points']:
        if point['status'] != 'complete':
            raise ValueError('incomplete campaign point')
        phases = {}
        for phase, data in point['phases'].items():
            if data['errors_skips_cancelled']:
                raise ValueError('erroneous request in campaign phase')
            totals = {'requests': 0, 'prompt_tokens': 0, 'output_tokens': 0, 'cached_tokens': 0}
            seen = set()
            for row in data['source_counts']:
                key = (row['trace'], row['row'])
                if key in seen:
                    raise ValueError('duplicate source aggregate')
                seen.add(key)
                prompt, output = expected[key]
                if phase == 'warmup':
                    output = 1
                count = row['count']
                if count <= 0 or row['prompt_tokens'] != count*prompt or row['output_tokens'] != count*output:
                    raise ValueError(f'usage length mismatch: {point["point"]["name"]} {phase} {key}')
                if not 0 <= row['cached_tokens'] <= row['prompt_tokens']:
                    raise ValueError('invalid cached token count')
                totals['requests'] += count
                for field in ['prompt_tokens', 'output_tokens', 'cached_tokens']:
                    totals[field] += row[field]
            compute = data['compute_tokens']
            if totals['requests'] != data['successful'] or any(totals[field] != compute[target] for field, target in [('prompt_tokens', 'input_tokens'), ('output_tokens', 'output_tokens'), ('cached_tokens', 'cached_tokens')]):
                raise ValueError('phase totals differ from source aggregates')
            phases[phase] = totals
        checked.append({'point': point['point']['name'], 'phases': phases})
    return {'valid': True, 'source_requests': len(expected), 'checked_points': len(checked), 'points': checked,
            'scope': 'usage input/output lengths and cached-count bounds checked against each frozen source row; one output token during warmup; this is not a token-ID equality check'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit_usage(json.loads(args.summary.read_text()), args.source), indent=2))
