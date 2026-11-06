from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
import urllib.request
from pathlib import Path


def request_json(url: str, endpoint: str, payload: dict | None = None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url + endpoint, data, {'Content-Type':'application/json'})
    with urllib.request.urlopen(request, timeout=600) as response:
        if endpoint == '/flush_cache':
            body = response.read().decode()
            if response.status != 200 or not body.startswith('Cache flushed.'):
                raise RuntimeError(body)
            return True
        return json.load(response)


def measure_overhead(url: str, source: Path, output: Path) -> dict:
    candidates = [(path.name, index, row) for path in sorted(source.glob('*.json')) for index, row in enumerate(json.loads(path.read_text())['requests'])]
    selected = random.Random(42).sample(candidates, 32)
    report = {'seed':42, 'selected': [{'file':p,'row':i,'input_ids':r['input_ids'],'output_length':len(r['output_ids'])} for p,i,r in selected], 'conditions':[], 'scope':'HTTP request wall time including JSON serialization, normal forward/sampling, staging and replacement; flush excluded equally; identical output IDs and cache work checked for every batch; not AgentX throughput'}
    for concurrency in [1,4,8]:
        batches = [selected[i:i+concurrency] for i in range(0,len(selected),concurrency)]
        reference = []
        for batch in batches:
            assert request_json(url, '/flush_cache')
            params = [{'temperature':0,'sampling_seed':42,'ignore_eos':True,'max_new_tokens':len(row['output_ids'])} for _,_,row in batch]
            reference.append(request_json(url, '/generate', {'input_ids':[row['input_ids'] for _,_,row in batch], 'sampling_params':params}))
        paired = []
        condition = {'concurrency':concurrency, 'pairs':paired, 'reference':reference}
        report['conditions'].append(condition)
        for repeat in range(5):
            pair = {'repeat':repeat,'arms':{}}
            for mode in (['generated','recorded'] if repeat%2==0 else ['recorded','generated']):
                wall = 0.0
                details = []
                for batch, expected in zip(batches,reference,strict=True):
                    assert request_json(url, '/flush_cache')
                    params = [{'temperature':0,'sampling_seed':42,'ignore_eos':True,'max_new_tokens':len(row['output_ids'])} for _,_,row in batch]
                    if mode=='recorded':
                        for param, result in zip(params,expected,strict=True):
                            param['replay_token_ids']=result['output_ids']
                    started=time.perf_counter()
                    results=request_json(url,'/generate',{'input_ids':[row['input_ids'] for _,_,row in batch],'sampling_params':params})
                    elapsed=time.perf_counter()-started
                    wall+=elapsed
                    for result, original in zip(results,expected,strict=True):
                        assert result['output_ids']==original['output_ids'], (concurrency,repeat,mode,'output drift')
                        assert result['meta_info']['finish_reason']['type']=='length'
                        assert result['meta_info']['cached_tokens']==original['meta_info']['cached_tokens']
                        assert result['meta_info']['prompt_tokens']==original['meta_info']['prompt_tokens']
                    details.append({'wall_seconds':elapsed,'prompt_tokens':sum(r['meta_info']['prompt_tokens'] for r in results),'cached_tokens':sum(r['meta_info']['cached_tokens'] for r in results),'output_tokens':sum(len(r['output_ids']) for r in results)})
                pair['arms'][mode]={'wall_seconds':wall,'batches':details}
            pair['overhead_fraction']=pair['arms']['recorded']['wall_seconds']/pair['arms']['generated']['wall_seconds']-1
            paired.append(pair)
            if len(paired) in (3, 5):
                values=[p['overhead_fraction'] for p in paired]
                mean=statistics.mean(values)
                critical={3:6.205347,5:3.495405}[len(values)]
                half=critical*statistics.stdev(values)/math.sqrt(len(values))
                condition['estimate']={'mean':mean,'ci95':[mean-half,mean+half],'pairs':len(values),'method':'paired relative wall-time mean; Bonferroni-adjusted Student t intervals for looks at n=3 and n=5; 95% simultaneous nominal coverage under t assumptions','target_met':mean+half<=0.03}
            output.write_text(json.dumps(report,indent=2))
            print(json.dumps({'concurrency':concurrency,'repeat':repeat,'overhead':pair['overhead_fraction'],'estimate':condition.get('estimate')}),flush=True)
            if len(paired)==3 and not condition['estimate']['ci95'][0]<=0.03<=condition['estimate']['ci95'][1]:
                break
    report['complete']=True
    report['target_met']=all(c['estimate']['target_met'] for c in report['conditions'])
    output.write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    measure_overhead(args.url,args.source,args.output)
