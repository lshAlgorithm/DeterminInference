from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path


def gpu_guard(indices: list[int]) -> str:
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    found = set()
    for line in result.splitlines():
        index, used, _, _ = map(int, line.split(','))
        if index in indices:
            found.add(index)
            if used >= 500:
                raise RuntimeError(f'GPU {index} occupied: {used} MiB')
    if found != set(indices):
        raise RuntimeError(f'missing GPUs: {indices}, found {found}')
    return result


def stop_process(process: subprocess.Popen | None) -> None:
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_point(spec: dict, point: dict, output: Path) -> dict:
    directory = output / point['name']
    directory.mkdir(parents=True, exist_ok=False)
    server = client = None
    receipt = {'point': point, 'seed': 42, 'status': 'starting'}
    server_cmd = list(spec['server']['cmd'])
    server_cmd[server_cmd.index('--max-total-tokens') + 1] = str(point['capacity'])
    client_env = {**os.environ, **spec['client']['env'], 'CONC': str(point['concurrency']), 'DURATION': str(point['duration']), 'RESULT_DIR': str(directory), 'AGENTIC_OUTPUT_DIR': str(directory), 'AGENTX_REPLAY_MODE': point['mode'], 'AGENTX_TOKEN_TRACE': point['trace']}
    receipt['server'] = {'cmd': server_cmd, 'env': spec['server']['env']}
    receipt['client'] = {'cmd': spec['client']['cmd'], 'env': {k:client_env[k] for k in set(spec['client']['env']) | {'CONC','DURATION','RESULT_DIR','AGENTIC_OUTPUT_DIR','AGENTX_REPLAY_MODE','AGENTX_TOKEN_TRACE'}}}
    url = spec['client']['env']['AIPERF_SERVER_URL']
    try:
        receipt['gpu_guard'] = gpu_guard(spec['gpu_indices'])
        started = time.monotonic()
        with (directory / 'server.log').open('x') as log:
            server = subprocess.Popen(server_cmd, env={**os.environ, **spec['server']['env']}, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt['server_pid'] = server.pid
        while True:
            if server.poll() is not None:
                raise RuntimeError(f'server exited {server.returncode}')
            try:
                with urllib.request.urlopen(url + '/health', timeout=3) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() - started > 600:
                raise TimeoutError('server startup exceeded 600 seconds')
            time.sleep(2)
        receipt['initialization_seconds'] = time.monotonic() - started
        for endpoint, name in [('/get_server_info', 'server-info.json'), ('/metrics', 'metrics-before.txt')]:
            with urllib.request.urlopen(url + endpoint, timeout=30) as response:
                (directory / name).write_bytes(response.read())
        receipt['status'] = 'profiling'
        (directory / 'receipt.json').write_text(json.dumps(receipt, indent=2))
        started = time.monotonic()
        with (directory / 'client.log').open('x') as log:
            client = subprocess.Popen(spec['client']['cmd'], env=client_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        with (directory / 'gpu.jsonl').open('x') as memory:
            while client.poll() is None:
                snapshot = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu,power.draw', '--format=csv,noheader,nounits'], text=True)
                memory.write(json.dumps({'elapsed_seconds': time.monotonic()-started, 'gpu': snapshot})+'\n')
                memory.flush()
                if server.poll() is not None:
                    raise RuntimeError(f'server exited during benchmark {server.returncode}')
                if time.monotonic() - started > point['duration'] + 1500:
                    raise TimeoutError('benchmark exceeded duration plus startup/grace allowance')
                time.sleep(5)
        receipt['client_wall_seconds'] = time.monotonic() - started
        receipt['client_returncode'] = client.returncode
        with urllib.request.urlopen(url + '/metrics', timeout=30) as response:
            (directory / 'metrics-after.txt').write_bytes(response.read())
        if client.returncode:
            raise RuntimeError(f'benchmark exited {client.returncode}')
        if spec.get('native_result'):
            native = json.loads((directory / spec['native_result']).read_text())
            if not native['complete']:
                raise RuntimeError('native measurement did not complete')
        else:
            records = [json.loads(line) for line in (directory/'aiperf_artifacts/profile_export.jsonl').open()]
            errors = [row for row in records if row.get('error') or row['metadata'].get('context_overflow_skip') or row['metadata'].get('was_cancelled')]
            receipt['records'] = len(records)
            receipt['errors'] = len(errors)
            if errors:
                raise RuntimeError(f'{len(errors)} request errors/skips/cancellations; inspect raw records')
        receipt['status'] = 'complete'
    except BaseException as error:
        receipt.update(status='failed', error=repr(error))
        raise
    finally:
        stop_process(client)
        stop_process(server)
        (directory / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'spec.json').write_text(json.dumps(spec, indent=2))
    state = {'status': 'running', 'pid': os.getpid(), 'completed': []}
    try:
        for point in spec['points']:
            state['current'] = point['name']
            (args.output / 'status.json').write_text(json.dumps(state, indent=2))
            result = run_point(spec, point, args.output)
            state['completed'].append({'name': point['name'], 'status': result['status']})
            print(json.dumps(state), flush=True)
            time.sleep(5)
        state['status'] = 'complete'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        (args.output / 'status.json').write_text(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
