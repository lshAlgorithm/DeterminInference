from __future__ import annotations

import argparse
import json
from pathlib import Path

from measure_overhead import request_json


def audit_dense_corpus(url: str, source: Path, output: Path) -> dict:
    rows=[]
    for path in sorted(source.glob('*.json')):
        trace=json.loads(path.read_text())
        request_json(url,'/flush_cache')
        for index,row in enumerate(trace['requests']):
            result=request_json(url,'/generate',{'input_ids':row['input_ids'],'extra_key':'audit-'+trace['id'],'sampling_params':{'temperature':0.7,'sampling_seed':42,'ignore_eos':True,'max_new_tokens':len(row['output_ids']),'replay_token_ids':row['output_ids']}})
            assert result['output_ids']==row['output_ids'], (path.name,index,result)
            assert result['meta_info']['finish_reason']['type']=='length'
            rows.append({'session':path.name,'request':index,'output_ids':result['output_ids'],'meta_info':result['meta_info']})
        print(json.dumps({'session':path.name,'requests_exact':len(rows)}),flush=True)
        output.write_text(json.dumps({'complete':False,'rows':rows},indent=2))
    result={'complete':True,'seed':42,'source':str(source),'requests_exact':len(rows),'rows':rows,'scope':'all qualified Dense requests executed from session start with exact output checks; reference execution, not an independent performance baseline'}
    output.write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    audit_dense_corpus(args.url,args.source,args.output)
