from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def compare_dense(summary: dict) -> dict:
    cells=defaultdict(dict)
    for result in summary['points']:
        point=result['point']
        cells[(point['capacity'],point['concurrency'],point['repeat'])][point['mode']]=result
    pairs=[]
    for (capacity,concurrency,repeat),arms in sorted(cells.items()):
        if set(arms)!={'generated','recorded'}:
            continue
        base=arms['generated']['phases']['profiling']
        replay=arms['recorded']['phases']['profiling']
        sources=[{(r['trace'],r['row']):r for r in arm['source_counts']} for arm in (base,replay)]
        common=sources[0].keys() & sources[1].keys()
        standardized=[]
        for mapping in sources:
            prompt=sum(mapping[key]['prompt_tokens']/mapping[key]['count'] for key in common)
            cached=sum(mapping[key]['cached_tokens']/mapping[key]['count'] for key in common)
            standardized.append({'common_source_rows':len(common),'prompt_tokens_once_per_row':prompt,'cached_tokens_once_per_row':cached,'cache_hit_fraction':cached/prompt if prompt else None})
        pairs.append({'capacity':capacity,'concurrency':concurrency,'repeat':repeat,'output_tps_relative_change':replay['output_tokens_per_second']/base['output_tokens_per_second']-1,'cache_hit_fraction_difference':replay['cache_hit_fraction']-base['cache_hit_fraction'],'source_standardized_cache_hit_fraction_difference':standardized[1]['cache_hit_fraction']-standardized[0]['cache_hit_fraction'] if common else None,'source_standardized':dict(zip(['generated','recorded'],standardized)),'actual_profile_counts':{'generated':base['successful'],'recorded':replay['successful']},'actual_ttft_mean_ms':{'generated':base['metrics']['time_to_first_token']['mean'],'recorded':replay['metrics']['time_to_first_token']['mean']}})
    groups=defaultdict(list)
    for pair in pairs:groups[(pair['capacity'],pair['concurrency'])].append(pair)
    estimates=[]
    for (capacity,concurrency),values in sorted(groups.items()):
        row={'capacity':capacity,'concurrency':concurrency,'pairs':len(values)}
        for metric in ['output_tps_relative_change','cache_hit_fraction_difference','source_standardized_cache_hit_fraction_difference']:
            data=[v[metric] for v in values if v[metric] is not None]
            mean=statistics.mean(data) if data else None
            half=4.302653*statistics.stdev(data)/math.sqrt(3) if len(data)==3 else None
            row[metric]={'pairs':len(data),'mean':mean,'ci95':None if half is None else [mean-half,mean+half]}
        estimates.append(row)
    return {'paired_results':pairs,'estimates':estimates,'complete_36_points':len(summary['points'])==36 and len(pairs)==18 and all(r['status']=='complete' for r in summary['points']),'scope':'closed-loop900s AgentX admission; throughput denominator includes actual drain; per-source standardization uses rows common to each pair, equally once per source row and weighted by prompt tokens; three-pair t intervals describe these runs, not source-population generalization'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(compare_dense(json.loads(args.summary.read_text())),indent=2))
