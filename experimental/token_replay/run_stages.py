from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    spec=json.loads(args.spec.read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'spec.json').write_text(json.dumps(spec,indent=2))
    state={'status':'waiting','pid':os.getpid(),'completed':[]}
    try:
        (args.output/'status.json').write_text(json.dumps(state,indent=2))
        dependency=Path(spec['wait_for'])
        while True:
            if not dependency.exists():
                time.sleep(10)
                continue
            previous=json.loads(dependency.read_text())
            if previous['status']=='failed':
                raise RuntimeError(f'prerequisite failed: {dependency}')
            if previous['status']=='complete':
                break
            time.sleep(10)
        for stage in spec['stages']:
            time.sleep(5)
            state.update(status='running',current=stage['name'])
            (args.output/'status.json').write_text(json.dumps(state,indent=2))
            with (args.output/(stage['name']+'.log')).open('x') as log:
                completed=subprocess.run(stage['cmd'],env={**os.environ,**stage['env']},stdout=log,stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f'{stage["name"]} exited {completed.returncode}')
            state['completed'].append(stage['name'])
        state['status']='complete'
    except BaseException as error:
        state.update(status='failed',error=repr(error))
        raise
    finally:
        (args.output/'status.json').write_text(json.dumps(state,indent=2))


if __name__=='__main__':
    main()
