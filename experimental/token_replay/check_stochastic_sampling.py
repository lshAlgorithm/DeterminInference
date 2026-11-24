from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from run_campaign import gpu_guard


def check_sampling(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    result = {'seed': 42, 'gpu_guard': gpu_guard([0]), 'status': 'starting'}
    try:
        import torch
        import flashinfer

        torch.manual_seed(42)
        probs = torch.softmax(torch.randn(64, 4096, device='cuda'), dim=-1)
        samples = []
        started = time.perf_counter()
        for _ in range(2):
            generator = torch.Generator(device='cuda').manual_seed(42)
            ids = flashinfer.sampling.top_k_top_p_sampling_from_probs(
                probs, top_k=50, top_p=0.9, generator=generator
            )
            torch.cuda.synchronize()
            assert ids.shape == (64,)
            assert bool(((ids >= 0) & (ids < 4096)).all())
            assert bool((probs.topk(50).indices == ids[:, None]).any(dim=1).all())
            samples.append(ids.cpu().tolist())
        assert samples[0] == samples[1]
        result.update(status='complete', samples=samples, elapsed_seconds=time.perf_counter()-started,
                      torch=torch.__version__, flashinfer=flashinfer.__version__)
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        (output/'receipt.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(check_sampling(parser.parse_args().output)))
