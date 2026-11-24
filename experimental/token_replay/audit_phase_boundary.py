from __future__ import annotations

import argparse
import json
from pathlib import Path


def audit_phase_boundary(records_path: Path) -> dict:
    phases = {'warmup': [], 'profiling': []}
    seen = set()
    for line in records_path.open():
        row = json.loads(line)
        metadata = row['metadata']
        phase = metadata['benchmark_phase']
        if phase not in phases:
            raise ValueError(f'unexpected phase: {phase}')
        identity = metadata['x_request_id']
        if identity in seen:
            raise ValueError('duplicate request ID')
        seen.add(identity)
        if row.get('error') or metadata.get('was_cancelled') or metadata.get('context_overflow_skip'):
            raise ValueError('failed request cannot establish a clean phase boundary')
        start, end = metadata['request_start_ns'], metadata['request_end_ns']
        if type(start) is not int or type(end) is not int or not 0 < start <= end:
            raise ValueError('invalid request interval')
        phases[phase].append((start, end))
    if not all(phases.values()):
        raise ValueError('both warmup and profiling requests are required')
    last_warmup_end = max(end for _, end in phases['warmup'])
    first_profile_start = min(start for start, _ in phases['profiling'])
    after_boundary = sum(end > first_profile_start for _, end in phases['warmup'])
    return {
        'records': str(records_path),
        'warmup_requests': len(phases['warmup']),
        'profiling_requests': len(phases['profiling']),
        'last_warmup_end_ns': last_warmup_end,
        'first_profile_start_ns': first_profile_start,
        'boundary_gap_ms': (first_profile_start-last_warmup_end)/1e6,
        'warmup_requests_ending_after_first_profile_start': after_boundary,
        'all_warmup_completed_before_profiling': after_boundary == 0,
        'scope': 'recorded client HTTP request intervals; does not measure GPU event overlap, remove warmup work, or equate snapshot state with complete historical replay',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audits = [audit_phase_boundary(path) for path in args.records]
    args.output.write_text(json.dumps({'runs': audits, 'all_boundaries_ordered': all(row['all_warmup_completed_before_profiling'] for row in audits)}, indent=2))
