from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from summarize_moe import summarize_service


def compare_recorder(root: Path) -> dict:
    pairs = []
    metrics = ['service_seconds', 'service_output_tps', 'mean_ttft_seconds', 'token_weighted_tpot_seconds']
    for repeat in range(3):
        dirs = {mode: root/f'moe-perturbation-r{repeat}-{mode}-01' for mode in ['off', 'on']}
        states = {mode: json.loads((path/'status.json').read_text()) for mode, path in dirs.items()}
        if any(state['status'] != 'complete' for state in states.values()):
            raise ValueError('perturbation arm incomplete')
        settings = []
        for mode, state in states.items():
            config = state['spec']['settings'].copy()
            recorder = config.pop('expert_distribution_recorder_mode')
            if recorder != ('per_token' if mode == 'on' else None):
                raise ValueError('unexpected recorder configuration')
            settings.append(config)
        if settings[0] != settings[1] or states['off']['spec']['sources'] != states['on']['spec']['sources']:
            raise ValueError('on/off model or source configuration mismatch')
        for content, source in states['off']['spec']['sources'].items():
            expected = {(p.name, i): [len(r['input_ids']), len(r['output_ids'])]
                        for p in sorted(Path(source).glob('*.json'))
                        for i, r in enumerate(json.loads(p.read_text())['requests'])}
            arms = {mode: summarize_service(path/content, expected) for mode, path in dirs.items()}
            for mode, path in dirs.items():
                summary = json.loads((path/content/'summary.json').read_text())
                if summary['record'] != (mode == 'on'):
                    raise ValueError('recorder run activation mismatch')
            if arms['off']['work_by_request'] != arms['on']['work_by_request']:
                raise ValueError('on/off token or cached-work mismatch')
            pairs.append({'repeat': repeat, 'content': content, 'arms': arms,
                          'initialization_seconds': {m: s['initialization_seconds'] for m, s in states.items()},
                          'on_relative_to_off': {m: arms['on'][m]/arms['off'][m]-1 for m in metrics}})
    estimates = {}
    for content in ['real', 'synthetic']:
        estimates[content] = {}
        for metric in metrics:
            values = [p['on_relative_to_off'][metric] for p in pairs if p['content'] == content]
            if len(values) != 3:
                raise ValueError('expected three pairs for each content')
            mean = statistics.mean(values)
            half = 4.302652729911275*statistics.stdev(values)/math.sqrt(3)
            estimates[content][metric] = {'mean': mean, 'ci95': [mean-half, mean+half]}
    return {'complete': True, 'pairs': pairs, 'estimates': estimates,
            'scope': 'three alternating recorder-on/off engine pairs per fixed content workload; full matched input/output/cache work; service timing excludes session dump, cache flush, startup and file I/O; initialization separately recorded; repeated-run t intervals, not population intervals'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(compare_recorder(args.root), indent=2))
