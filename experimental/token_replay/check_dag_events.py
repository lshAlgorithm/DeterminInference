from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def check_dag_events(records: Path, server_log: Path) -> dict:
    wire = {}
    for line in server_log.open():
        if line.startswith('{'):
            event = json.loads(line)
            if event.get('event') == 'request.received.openai':
                wire[event['headers']['x-request-id']] = event['obj']
    trees = defaultdict(list)
    failures = []
    for line in records.open():
        meta = json.loads(line)['metadata']
        meta['salt'] = wire[meta['x_request_id']]['cache_salt']
        trees[meta['root_correlation_id']].append(meta)
    salts = {}
    counts = defaultdict(int)
    child_idle = []
    for root, rows in trees.items():
        tree_salts = {row['salt'] for row in rows}
        if len(tree_salts) != 1:
            failures.append({'kind': 'tree_salt', 'root': root})
        salt = rows[0]['salt']
        if salt in salts:
            failures.append({'kind': 'independent_tree_salt', 'root': root})
        salts[salt] = root
        phases = {row['benchmark_phase'] for row in rows}
        counts['warmup_to_profile_trees'] += phases == {'warmup', 'profiling'}
        child = sorted((r for r in rows if r['source_kind'] == 'weka_subagent'), key=lambda r:r['request_start_ns'])
        joined = [r for r in rows if r['source_outer_idx'] == 2]
        reset = [r for r in rows if r['source_outer_idx'] == 3]
        for parent in joined:
            prior = [c for c in child if c['request_start_ns'] < parent['request_start_ns']]
            if prior:
                counts['joins'] += 1
                if max(c['request_end_ns'] for c in prior) > parent['request_start_ns']:
                    failures.append({'kind': 'join_before_child_end', 'root': root})
        for previous, current in zip(child, child[1:]):
            if current['benchmark_phase'] == previous['benchmark_phase'] == 'profiling':
                gap = (current['request_start_ns'] - previous['request_end_ns']) / 1e9
                child_idle.append(gap)
                if gap < 0.74:
                    failures.append({'kind': 'missing_child_idle', 'gap': gap})
        counts['reset_trees'] += bool(reset)
        if joined and reset:
            if min(r['request_start_ns'] for r in reset) < max(r['request_end_ns'] for r in joined):
                failures.append({'kind': 'reset_before_join', 'root': root})
        counts['from_start_trees'] += any(r['source_outer_idx'] == 0 and r['benchmark_phase'] == 'profiling' for r in rows)
    for key in ['warmup_to_profile_trees', 'joins', 'reset_trees', 'from_start_trees']:
        if not counts[key]:
            failures.append({'kind': 'missing_coverage', 'coverage': key})
    if not child_idle:
        failures.append({'kind': 'missing_idle_coverage'})
    return {'passed': not failures, 'failures': failures, 'trees': len(trees), 'counts': dict(counts), 'child_idle_seconds': child_idle, 'scope': 'derived DAG event order and actual wire salt mapping; cache isolation separately witnessed'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--server-log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = check_dag_events(args.records, args.server_log)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
