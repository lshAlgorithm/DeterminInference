from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def audit_recorder(directory: Path, source: Path, model_config: Path) -> dict:
    import torch

    config = json.loads(model_config.read_text())
    layers, experts, topk = (config[k] for k in ['num_hidden_layers', 'num_experts', 'num_experts_per_tok'])
    sources = sorted(source.glob('*.json'))
    rows_list = [json.loads(line) for line in (directory/'requests.jsonl').read_text().splitlines()]
    rows = {(r['session'], r['request']): r for r in rows_list}
    assert sources and len(rows) == len(rows_list)
    reports = []
    totals = {}
    expected_keys = set()
    for rank in range(2):
        files = sorted(directory.glob(f'expert_distribution_recorder_*_{rank}.pt'), key=lambda p: float(p.stem.split('_')[-2]))
        assert len(files) == len(sources), (rank, len(files), len(sources))
        total = {phase: torch.zeros((layers, experts), dtype=torch.int64) for phase in ['prefill', 'decode']}
        for path, session in zip(files, sources, strict=True):
            payload = torch.load(path, map_location='cpu', weights_only=False)
            assert torch.equal(payload['last_physical_to_logical_map'], torch.arange(experts).repeat(layers, 1))
            segments = []
            for record in payload['records']:
                mode = record['forward_mode']
                assert mode in (1, 2), mode
                if mode == 1 and (not segments or segments[-1]['decode']):
                    segments.append({'prefill': [], 'decode': []})
                phase = 'decode' if mode == 2 else 'prefill'
                segments[-1][phase].append(record)
                assert record['rank'] == rank
                hist = record['global_physical_count'].to(torch.int64)
                assert hist.shape == (layers, experts) and bool((hist >= 0).all())
                assert torch.equal(hist.sum(1), torch.full((layers,), topk*len(record['input_ids'])))
                total[phase] += hist
            requests = json.loads(session.read_text())['requests']
            assert len(segments) == len(requests), (session.name, len(segments), len(requests))
            counts = {'prefill': 0, 'decode': 0}
            for index, (segment, request) in enumerate(zip(segments, requests, strict=True)):
                key = (session.name, index)
                expected_keys.add(key)
                row = rows[key]
                assert row['output_ids_exact']
                for phase in ['prefill', 'decode']:
                    ids = [v for r in segment[phase] for v in r['input_ids']]
                    positions = [v for r in segment[phase] for v in r['positions']]
                    start = row['cached_tokens'] if phase == 'prefill' else len(request['input_ids'])
                    expected = request['input_ids'][start:] if phase == 'prefill' else request['output_ids']
                    assert ids == expected, (rank, session.name, index, phase, 'IDs')
                    assert positions == list(range(start, start+len(expected))), (rank, session.name, index, phase, 'positions')
                    counts[phase] += len(ids)
            reports.append({'rank': rank, 'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'session': session.name, 'requests': len(requests), 'forward_input_tokens': counts})
            print(json.dumps(reports[-1]), flush=True)
        totals[rank] = total
    assert set(rows) == expected_keys
    assert all(torch.equal(totals[0][phase], totals[1][phase]) for phase in totals[0])
    return {'complete': True, 'sessions': len(sources), 'requests': len(expected_keys), 'rank_files': reports,
            'selected_rank': 0, 'per_layer_expert_counts': {phase: h.tolist() for phase, h in totals[0].items()},
            'model_config_sha256': hashlib.sha256(model_config.read_bytes()).hexdigest(),
            'scope': 'exact raw recorder input/position coverage for fixed sequential overlap workload on both ranks; full output length decode forwards include final discarded overlap result; counts are not GPU execution time'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--model-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit_recorder(args.directory, args.source, args.model_config), indent=2))
