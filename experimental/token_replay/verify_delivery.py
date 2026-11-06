from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from audit_dense_usage import audit_usage


def verify_delivery(root: Path) -> dict:
    summary = json.loads((root/'dense-formal-01-summary.json').read_text())
    comparison = json.loads((root/'dense-formal-01-comparison.json').read_text())
    expected_cells = {(capacity, concurrency, repeat, mode) for capacity in [131072, 16384] for concurrency in [1, 4, 8] for repeat in range(3) for mode in ['generated', 'recorded']}
    actual_cells = [(r['point']['capacity'], r['point']['concurrency'], r['point']['repeat'], r['point']['mode']) for r in summary['points']]
    if len(actual_cells) != 36 or set(actual_cells) != expected_cells or not comparison['complete_36_points']:
        raise ValueError('formal Dense grid incomplete')
    for row in summary['points']:
        point = row['point']
        if point['duration'] != 900 or row['original_aggregate']['submission_valid'] is not False:
            raise ValueError('formal duration or submission provenance mismatch')
        receipt = json.loads((root/'dense-formal-01'/point['name']/'receipt.json').read_text())
        if receipt['status'] != 'complete' or receipt['errors'] or receipt['point'] != point:
            raise ValueError('invalid campaign receipt')
        command = receipt['server']['cmd']
        if int(command[command.index('--max-total-tokens')+1]) != point['capacity'] or command[command.index('--context-length')+1] != '16384':
            raise ValueError('serving capacity or context mismatch')
        env = receipt['client']['env']
        if env['CONC'] != str(point['concurrency']) or env['DURATION'] != '900' or env['AGENTX_REPLAY_MODE'] != point['mode']:
            raise ValueError('client replay condition mismatch')
    usage = audit_usage(summary, root/'exgentic-dense-02/sessions')
    expected = {(p.name, i): r for p in sorted((root/'exgentic-dense-02/sessions').glob('*.json')) for i, r in enumerate(json.loads(p.read_text())['requests'])}
    reference = json.loads((root/'dense-corpus-audit-01/reference/audit.json').read_text())
    if not reference['complete'] or len(expected) != 447 or reference['requests_exact'] != 447:
        raise ValueError('Dense all-corpus reference incomplete')
    seen = set()
    for row in reference['rows']:
        key = (row['session'], row['request'])
        if key in seen or row['output_ids'] != expected[key]['output_ids']:
            raise ValueError('duplicate or inexact corpus reference output')
        seen.add(key)
    if seen != set(expected):
        raise ValueError('missing Dense reference requests')
    overhead = json.loads((root/'dense-overhead-02/matched/overhead.json').read_text())
    if not overhead['complete'] or not overhead['target_met']:
        raise ValueError('scoped HTTP overhead target not established')
    moe = json.loads((root/'moe-service-routing-summary.json').read_text())
    recorder = json.loads((root/'moe-recorder-paired-summary.json').read_text())
    ranks = json.loads((root/'moe-rank-comparison.json').read_text())
    if not (moe['complete_off'] and moe['complete_on'] and recorder['complete'] and ranks['complete']):
        raise ValueError('MoE evidence incomplete')
    for content in ['real', 'synthetic']:
        audit = moe['routing'][content]['coverage_audit']
        if not audit['complete'] or audit['requests'] != 308 or audit['sessions'] != 21:
            raise ValueError('MoE routing coverage incomplete')
    if ranks['representative_requests_per_content'] != 9:
        raise ValueError('representative rank coverage mismatch')
    public = json.loads((root/'weka-original-04-summary.json').read_text())['points'][0]
    if public['status'] != 'complete' or any(p['errors_skips_cancelled'] for p in public['phases'].values()):
        raise ValueError('public reference incomplete')
    storage = json.loads((root/'storage-qualified-01/storage.json').read_text())
    if not storage['complete'] or len(storage['records']) != 44 or any(r['allocated_after_release_delta_bytes'] for r in storage['records']):
        raise ValueError('storage allocation diagnostic incomplete')
    return {'deterministic_completion_checks': 'passed', 'formal_points': 36, 'dense_usage': usage,
            'dense_reference_exact_requests': 447, 'moe_routing_exact_requests_per_content': 308,
            'moe_rank_requests_per_content': 9, 'recorder_pairs_per_content': 3,
            'overhead_target_scope': 'fixed32-prompt matched-output greedy HTTP experiment, not a source-population guarantee',
            'review': {'codex': 'provisional independent reviews', 'claude': 'unavailable; no dual-review acceptance'},
            'validator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope': 'checks delivery denominators, exact-ID reference outputs and explicit completion gates; does not establish a positive performance effect or universal low-overhead guarantee'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(verify_delivery(args.root), indent=2))
