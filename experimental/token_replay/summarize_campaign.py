from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def summarize_point(directory: Path) -> dict:
    receipt = json.loads((directory/'receipt.json').read_text())
    records = [json.loads(line) for line in (directory/'aiperf_artifacts/profile_export.jsonl').open()]
    exported = json.loads((directory/'aiperf_artifacts/server_metrics_export.json').read_text())
    aggregate = json.loads((directory/'entry.json').read_text())
    phases = {}
    for phase in ['warmup','profiling']:
        rows = [r for r in records if r['metadata']['benchmark_phase']==phase]
        valid = [r for r in rows if not r.get('error') and not r['metadata'].get('was_cancelled') and not r['metadata'].get('context_overflow_skip')]
        counters = Counter()
        distributions = defaultdict(list)
        sources = Counter()
        for row in valid:
            values = {k:v['value'] for k,v in row['metrics'].items()}
            counters['input_tokens'] += values.get('usage_prompt_tokens',values.get('input_sequence_length',0))
            counters['output_tokens'] += values.get('usage_completion_tokens',values.get('output_sequence_length',0))
            counters['cached_tokens'] += values.get('usage_prompt_cache_read_tokens',0)
            meta=row['metadata']
            sources[(meta['source_trace_id'], meta['source_outer_idx'])]+=1
            for key in ['time_to_first_token','inter_token_latency','request_latency','output_token_throughput_per_user','e2e_output_token_throughput']:
                if key in values:
                    distributions[key].append(values[key])
        counters['new_prefill_tokens'] = counters['input_tokens']-counters['cached_tokens']
        metric_set = exported['warmup_metrics'] if phase=='warmup' else exported['metrics']
        selected = {key:[{'labels':s['labels'],'stats':s['stats']} for s in value['series']] for key,value in metric_set.items() if any(part in key for part in ['evicted_tokens','retract','token_usage','cached_tokens','prompt_tokens','generation_tokens','num_requests'])}
        phase_range = exported['summary']['phase_time_ranges'][phase]
        seconds=(phase_range['end_ns']-phase_range['start_ns'])/1e9
        phases[phase]={'records':len(rows),'successful':len(valid),'errors_skips_cancelled':len(rows)-len(valid),'compute_tokens':dict(counters),'phase_seconds_including_drain':seconds,'output_tokens_per_second':counters['output_tokens']/seconds,'cache_hit_fraction':counters['cached_tokens']/counters['input_tokens'] if counters['input_tokens'] else None,'metrics':{key:{'n':len(v),'mean':statistics.mean(v),'median':statistics.median(v),'min':min(v),'max':max(v)} for key,v in distributions.items()},'source_counts':[{'trace':t,'row':i,'count':n} for (t,i),n in sorted(sources.items())],'server_metrics':selected}
    memory=[]
    for line in (directory/'gpu.jsonl').open():
        sample=json.loads(line)
        for row in sample['gpu'].splitlines():
            fields=row.split(',')
            if int(fields[0])==0:
                memory.append(float(fields[1]))
    return {'point':receipt['point'],'status':receipt['status'],'submission_valid':aggregate.get('submission_valid',False),'initialization_seconds':receipt['initialization_seconds'],'gpu0_peak_sampled_memory_mib':max(memory),'phases':phases,'original_aggregate':aggregate,'scope':'closed-loop AgentX throughput; actual request mix and denominator retained; cached tokens from response usage, eviction/retraction counters from phase-scoped server metrics'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rows=[]
    for directory in sorted(args.campaign.iterdir()):
        if (directory/'entry.json').exists():
            rows.append(summarize_point(directory))
    args.output.write_text(json.dumps({'points':rows},indent=2))
    print(json.dumps([{'point':r['point'],'status':r['status'],'profile_requests':r['phases']['profiling']['successful']} for r in rows],indent=2))
