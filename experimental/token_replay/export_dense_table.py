from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def export_table(summary: dict, output: Path) -> None:
    rows = []
    for result in summary['points']:
        point = result['point']
        for phase, data in result['phases'].items():
            row = {'point': point['name'], 'capacity_tokens': point['capacity'], 'concurrency': point['concurrency'],
                   'repeat': point['repeat'], 'mode': point['mode'], 'phase': phase, 'status': result['status'],
                   'requests': data['records'], 'successful': data['successful'], 'errors_skips_cancelled': data['errors_skips_cancelled'],
                   'phase_seconds_including_drain': data['phase_seconds_including_drain'],
                   'output_tokens_per_second_per_gpu': data['output_tokens_per_second'], 'cache_hit_fraction': data['cache_hit_fraction'],
                   'initialization_seconds': result['initialization_seconds'],
                   'gpu0_peak_sampled_memory_mib_during_client': result['gpu0_peak_sampled_memory_mib'], **data['compute_tokens']}
            for key in ['time_to_first_token', 'inter_token_latency', 'request_latency', 'output_token_throughput_per_user']:
                values = data['metrics'].get(key, {})
                row[key+'_n'] = values.get('n', 0)
                row[key+'_mean'+('_tokens_per_second' if key.endswith('per_user') else '_ms')] = values.get('mean')
            for metric, field, statistic in [('sglang:evicted_tokens', 'evicted_tokens_phase_total', 'total'), ('sglang:num_retractions', 'retractions_phase_total', 'sum')]:
                series = data['server_metrics'].get(metric)
                row[field] = sum(item['stats'][statistic] for item in series) if series else None
            residency = data['cache_residency']
            row['cache_residency_sample_count'] = residency['sample_count']
            metrics = residency['sampled_metrics'] or {}
            for field in ['prefix_resident_tokens', 'evictable_tokens', 'protected_prefix_tokens', 'allocated_kv_tokens']:
                row[field+'_sampled_max'] = metrics.get(field, {}).get('max')
            rows.append(row)
    if not rows:
        raise ValueError('no campaign rows to export')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export_table(json.loads(args.summary.read_text()), args.output)
