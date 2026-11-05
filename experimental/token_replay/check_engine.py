import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import sglang as sgl
import torch
from sglang.srt.sampling.token_replay import TokenReplayBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from transformers import AutoTokenizer


def check_validation() -> None:
    valid = {"max_new_tokens": 2, "ignore_eos": True, "replay_token_ids": [1, 2]}
    SamplingParams(**valid).verify(10)
    for change in (
        {"replay_token_ids": []},
        {"replay_token_ids": [1, 10]},
        {"replay_token_ids": [True, 2]},
        {"max_new_tokens": 3},
        {"ignore_eos": False},
        {"stop": ["end"]},
        {"regex": "a"},
        {"regex": ""},
    ):
        try:
            SamplingParams(**(valid | change)).verify(10)
        except ValueError:
            continue
        raise AssertionError(f"invalid replay accepted: {change}")


def check_completion_adapter() -> None:
    from sglang.srt.entrypoints.openai.protocol import CompletionRequest
    from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion

    request = CompletionRequest(
        prompt=[1, 2], max_tokens=2, ignore_eos=True, replay_token_ids=[8, 9],
    )
    adapter = object.__new__(OpenAIServingCompletion)
    params = adapter._build_sampling_params(request)
    assert params["replay_token_ids"] == [8, 9]
    SamplingParams(**params).verify(10)
    try:
        CompletionRequest(prompt=[1], replay_token_ids=[True])
    except ValueError:
        return
    raise AssertionError("completion protocol accepted a boolean token ID")


def check_kernel() -> None:
    requests = [
        SimpleNamespace(
            origin_input_ids=[1] * 4,
            sampling_params=SimpleNamespace(replay_token_ids=[11, 12, 13]),
        ),
        SimpleNamespace(
            origin_input_ids=[1] * 7,
            sampling_params=SimpleNamespace(replay_token_ids=None),
        ),
        SimpleNamespace(
            origin_input_ids=[1] * 5,
            sampling_params=SimpleNamespace(replay_token_ids=[21, 22]),
        ),
    ]
    batch = TokenReplayBatch.from_requests(requests, "cuda")
    sampled = torch.tensor([99, 98, 97], device="cuda", dtype=torch.int32)
    batch.apply(sampled, torch.tensor([3, 6, 5], device="cuda"))
    assert sampled.tolist() == [11, 98, 22]
    reordered = batch.filter(torch.tensor([2, 0], device="cuda"), [2, 0])
    merged = TokenReplayBatch.merge(None, reordered, 1, 2, "cuda")
    sampled = torch.tensor([90, 91, 92], device="cuda")
    merged.apply(sampled, torch.tensor([100, 4, 5], device="cuda"))
    assert sampled.tolist() == [90, 21, 13]
    reverse = TokenReplayBatch.merge(reordered, None, 2, 1, "cuda")
    sampled = torch.tensor([90, 91, 92], device="cuda")
    reverse.apply(sampled, torch.tensor([5, 4, 100], device="cuda"))
    assert sampled.tolist() == [22, 12, 92]


def check_engine(model: str) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model)
    prompt = tokenizer.encode("A short explanation of why the sky looks blue:\n")
    engine = sgl.Engine(
        model_path=model,
        skip_tokenizer_init=True,
        random_seed=42,
        mem_fraction_static=0.55,
        context_length=4096,
        max_running_requests=8,
        cuda_graph_max_bs=8,
        attention_backend="triton",
        log_level="warning",
    )
    try:
        ordinary = {"temperature": 0, "max_new_tokens": 24, "ignore_eos": True}
        recorded = engine.generate(input_ids=prompt, sampling_params=ordinary)[
            "output_ids"
        ]
        engine.flush_cache()
        forced = {**ordinary, "temperature": 0.9, "replay_token_ids": recorded}
        replayed = engine.generate(
            input_ids=prompt, sampling_params=forced, return_logprob=True
        )
        assert replayed["output_ids"] == recorded, replayed
        assert [
            x[1] for x in replayed["meta_info"]["output_token_logprobs"]
        ] == recorded

        followup = (
            prompt + recorded + tokenizer.encode("\nSummarize that in one sentence:\n")
        )
        cached = engine.generate(
            input_ids=followup, sampling_params=ordinary, return_logprob=True
        )
        assert (
            cached["meta_info"]["cached_tokens"] >= len(prompt) + len(recorded) - 1
        ), cached
        engine.flush_cache()
        fresh = engine.generate(
            input_ids=followup, sampling_params=ordinary, return_logprob=True
        )
        assert cached["output_ids"] == fresh["output_ids"], (cached, fresh)
        for a, b in zip(
            cached["meta_info"]["output_token_logprobs"],
            fresh["meta_info"]["output_token_logprobs"],
            strict=True,
        ):
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
            order = (
                ("ordinary", "replay") if iteration % 2 == 0 else ("replay", "ordinary")
            )
            for mode in order:
                engine.flush_cache()
                params = (
                    ordinary
                    if mode == "ordinary"
                    else {**ordinary, "replay_token_ids": recorded}
                )
                start = time.perf_counter()
                out = engine.generate(input_ids=prompt, sampling_params=params)
                elapsed = time.perf_counter() - start
                assert out["output_ids"] == recorded
                if iteration >= 2:
                    timings[mode].append(elapsed)
        return {
            "model": model,
            "seed": 42,
            "recorded_ids": recorded,
            "cached_tokens": cached["meta_info"]["cached_tokens"],
            "expected_cache_prefix": len(prompt) + len(recorded) - 1,
            "mixed_batch_and_eos": "passed",
            "cached_vs_fresh": "passed",
            "request_seconds_including_prefill": timings,
            "trace_rows": [
                {
                    "session_id": "sky",
                    "input_ids": prompt,
                    "output_ids": recorded,
                    "delay_seconds": 0,
                },
                {
                    "session_id": "sky",
                    "input_ids": followup,
                    "output_ids": fresh["output_ids"],
                    "delay_seconds": 0,
                },
            ],
        }
    finally:
        engine.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace-output", type=Path, required=True)
    args = parser.parse_args()
    check_validation()
    check_kernel()
    check_completion_adapter()
    report = check_engine(args.model)
    args.trace_output.write_text(
        "\n".join(json.dumps(row) for row in report.pop("trace_rows")) + "\n"
    )
    print(json.dumps(report, indent=2))
