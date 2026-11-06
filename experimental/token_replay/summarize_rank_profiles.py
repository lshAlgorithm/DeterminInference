from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path


def summarize_rank_profiles(directory: Path) -> dict:
    reports=[]
    for path in sorted(directory.rglob('*.trace.json*')):
        opener=gzip.open if path.suffix=='.gz' else open
        with opener(path,'rt') as stream:
            trace=json.load(stream)
        devices=defaultdict(list)
        for event in trace['traceEvents']:
            if event.get('cat')=='kernel' and event.get('ph')=='X':
                devices[str(event.get('args',{}).get('device','unknown'))].append(event)
        for device,events in devices.items():
            groups=defaultdict(list)
            names=defaultdict(float)
            for event in events:
                kind='communication' if re.search(r'nccl|all_?reduce|all_?gather|reduce_scatter|cross_device|broadcast',event['name'],re.I) else 'compute_and_other'
                groups[kind].append((event['ts'],event['ts']+event['dur']))
                names[event['name']]+=event['dur']/1000
            durations={}
            for kind,intervals in groups.items():
                merged=[]
                for start,end in sorted(intervals):
                    if merged and start<=merged[-1][1]:
                        merged[-1]=(merged[-1][0],max(end,merged[-1][1]))
                    else:
                        merged.append((start,end))
                durations[kind]={'kernel_sum_ms':sum(end-start for start,end in intervals)/1000,'union_ms':sum(end-start for start,end in merged)/1000,'kernel_count':len(intervals)}
            reports.append({'file':str(path.relative_to(directory)),'visible_device':device,'kernel_span_ms':(max(e['ts']+e['dur'] for e in events)-min(e['ts'] for e in events))/1000,'durations':durations,'kernel_names_ms':dict(sorted(names.items(),key=lambda item:-item[1]))})
    if not reports:
        raise ValueError('no CUDA kernel events found in rank traces')
    return {'profiles':reports,'scope':'CUDA kernel sums and within-category unions for frozen representative requests; category overlaps are not additive wall time; communication classified by retained kernel names; profiler-on diagnostic, not uninstrumented whole-workload timing'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(summarize_rank_profiles(args.directory),indent=2))
