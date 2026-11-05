import argparse
import json
import math
import time

import sglang as sgl
from transformers import AutoTokenizer


def check_engine(model: str) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model)
    prompt = tokenizer.encode("A short explanation of why the sky looks blue:\n")
    engine = sgl.Engine(
        model_path=model, skip_tokenizer_init=True, random_seed=42,
        mem_fraction_static=0.55, context_length=4096, max_running_requests=8,
        cuda_graph_max_bs=8, attention_backend="triton", log_level="warning",
    )
    try:
        ordinary = {"temperature": 0, "max_new_tokens": 24, "ignore_eos": True}
        recorded = engine.generate(input_ids=prompt, sampling_params=ordinary)["output_ids"]
        engine.flush_cache()
        forced = {**ordinary, "temperature": 0.9, "replay_token_ids": recorded}
        replayed = engine.generate(input_ids=prompt, sampling_params=forced, return_logprob=True)
        assert replayed["output_ids"] == recorded, replayed
        assert [x[1] for x in replayed["meta_info"]["output_token_logprobs"]] == recorded

        followup = prompt + recorded + tokenizer.encode("\nSummarize that in one sentence:\n")
        cached = engine.generate(input_ids=followup, sampling_params=ordinary, return_logprob=True)
        assert cached["meta_info"]["cached_tokens"] >= len(prompt) + len(recorded) - 1, cached
        engine.flush_cache()
        fresh = engine.generate(input_ids=followup, sampling_params=ordinary, return_logprob=True)
        assert cached["output_ids"] == fresh["output_ids"], (cached, fresh)
        for a, b in zip(cached["meta_info"]["output_token_logprobs"], fresh["meta_info"]["output_token_logprobs"], strict=True):
            assert math.isclose(a[0], b[0], abs_tol=0.05), (a, b)

        eos_trace = [recorded[0], tokenizer.eos_token_id, recorded[1]]
        mixed = engine.generate(
            input_ids=[prompt, prompt + [recorded[0]], followup],
            sampling_params=[
                {**ordinary, "max_new_tokens": 3, "replay_token_ids": eos_trace},
                {**ordinary, "max_new_tokens": 7, "replay_token_ids": recorded[:7]},
                {**ordinary, "max_new_tokens": 11},
            ],
        )
        assert mixed[0]["output_ids"] == eos_trace, mixed[0]
        assert mixed[1]["output_ids"] == recorded[:7], mixed[1]
        assert len(mixed[2]["output_ids"]) == 11, mixed[2]
        timings = {"ordinary": [], "replay": []}
        for iteration in range(8):
            order = ("ordinary", "replay") if iteration % 2 == 0 else ("replay", "ordinary")
            for mode in order:
                engine.flush_cache()
                params = ordinary if mode == "ordinary" else {**ordinary, "replay_token_ids": recorded}
                start = time.perf_counter()
                out = engine.generate(input_ids=prompt, sampling_params=params)
                elapsed = time.perf_counter() - start
                assert out["output_ids"] == recorded
                if iteration >= 2:
                    timings[mode].append(elapsed)
        return {
            "model": model, "seed": 42, "recorded_ids": recorded,
            "cached_tokens": cached["meta_info"]["cached_tokens"],
            "expected_cache_prefix": len(prompt) + len(recorded) - 1,
            "mixed_batch_and_eos": "passed", "cached_vs_fresh": "passed",
            "request_seconds_including_prefill": timings,
        }
    finally:
        engine.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    print(json.dumps(check_engine(args.model), indent=2))
