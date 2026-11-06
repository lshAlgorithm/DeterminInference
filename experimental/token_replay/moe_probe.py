from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import time
from pathlib import Path


def guard_gpus() -> list[dict]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
        raise ValueError("this diagnostic requires CUDA_VISIBLE_DEVICES=2,3")
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    devices = []
    for row in rows.splitlines():
        index, name, used, total, utilization = [
            part.strip() for part in row.split(",")
        ]
        if index in {"2", "3"}:
            device = {
                "index": int(index),
                "name": name,
                "used_mib": int(used),
                "total_mib": int(total),
                "utilization": int(utilization),
            }
            if device["used_mib"] >= 500:
                raise RuntimeError(f"GPU {index} is occupied: {device}")
            devices.append(device)
    if len(devices) != 2:
        raise RuntimeError("both physical GPUs 2 and 3 must be visible")
    return devices


def model_manifest(model: Path) -> dict:
    index = json.loads((model / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    files = []
    for name in [
        "config.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        *shards,
    ]:
        path = model / name
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(8 << 20):
                digest.update(chunk)
        files.append(
            {
                "name": name,
                "size_bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return {
        "model": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        "revision": "ec052fda178e241c7c443468d2fa1db6618996be",
        "declared_weight_bytes": index["metadata"]["total_size"],
        "files": files,
    }


def check_engine(model: Path, output: Path, record: bool, ep_size: int) -> dict:
    from transformers import AutoTokenizer

    import sglang as sgl

    tokenizer = AutoTokenizer.from_pretrained(model)
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Explain why prefix caching can accelerate a multi-turn agent.",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt = tokenizer.encode(prompt, add_special_tokens=False)
    settings = {
        "model_path": str(model),
        "dtype": "bfloat16",
        "tp_size": 2,
        "ep_size": ep_size,
        "random_seed": 42,
        "skip_tokenizer_init": True,
        "attention_backend": "triton",
        "moe_runner_backend": "triton",
        "moe_a2a_backend": "none",
        "enable_eplb": False,
        "init_expert_location": "trivial",
        "disable_shared_experts_fusion": True,
        "mem_fraction_static": 0.82,
        "context_length": 16384,
        "max_total_tokens": 16384,
        "max_running_requests": 8,
        "cuda_graph_max_bs": 8,
        "disable_piecewise_cuda_graph": True,
        "expert_distribution_recorder_mode": "per_token" if record else None,
        "log_level": "info",
    }
    started = time.perf_counter()
    engine = sgl.Engine(**settings)
    init_seconds = time.perf_counter() - started
    try:
        params = {
            "temperature": 0,
            "max_new_tokens": 32,
            "ignore_eos": True,
            "sampling_seed": 42,
        }
        generated = engine.generate(input_ids=prompt, sampling_params=params)
        recorded = generated["output_ids"]
        engine.flush_cache()
        if record:
            engine.start_expert_distribution_record()
        replayed = engine.generate(
            input_ids=prompt,
            sampling_params={
                **params,
                "temperature": 0.7,
                "replay_token_ids": recorded,
            },
            return_logprob=True,
        )
        if replayed["output_ids"] != recorded:
            raise AssertionError("MoE replay output IDs differ")
        if [
            entry[1] for entry in replayed["meta_info"]["output_token_logprobs"]
        ] != recorded:
            raise AssertionError("MoE logprob token IDs differ")
        followup = prompt + recorded + tokenizer.encode("\nSummarize in one sentence:")
        cached = engine.generate(input_ids=followup, sampling_params=params)
        if cached["meta_info"]["cached_tokens"] < len(prompt) + len(recorded) - 1:
            raise AssertionError("MoE followup failed expected prefix reuse")
        if record:
            engine.stop_expert_distribution_record()
            engine.dump_expert_distribution_record()
        return {
            "settings": settings,
            "initialization_seconds": init_seconds,
            "generated": generated,
            "replayed": replayed,
            "followup": cached,
            "server_info": engine.get_server_info(),
            "scope": "correctness feasibility fixture, not agent workload or performance evidence",
        }
    finally:
        engine.shutdown()


def summarize_recorder(directory: Path, ep_size: int) -> dict:
    import torch
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    summaries = []
    for path in sorted(directory.glob("expert_distribution_recorder_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        counts = {}
        for record in payload["records"]:
            mode = ForwardMode(record["forward_mode"])
            phase = (
                "decode"
                if mode.is_decode()
                else "prefill"
                if mode.is_prefill()
                else mode.name
            )
            histogram = record["global_physical_count"].to(torch.float64)
            counts[phase] = counts.get(phase, torch.zeros_like(histogram)) + histogram
        phases = {}
        for phase, histogram in counts.items():
            expert_mean = histogram.mean(dim=1)
            per_card = histogram.reshape(histogram.shape[0], ep_size, -1).sum(dim=2)
            card_mean = per_card.mean(dim=1)
            phases[phase] = {
                "per_layer_physical_expert_counts": histogram.to(torch.int64).tolist(),
                "per_layer_expert_max_over_mean": (
                    histogram.max(dim=1).values / expert_mean.clamp_min(1)
                ).tolist(),
                "per_layer_expert_cv": (
                    histogram.std(dim=1, correction=0) / expert_mean.clamp_min(1)
                ).tolist(),
                "per_layer_trivial_placement_card_token_load": per_card.to(
                    torch.int64
                ).tolist(),
                "per_layer_card_max_over_mean": (
                    per_card.max(dim=1).values / card_mean.clamp_min(1)
                ).tolist(),
                "per_layer_card_cv": (
                    per_card.std(dim=1, correction=0) / card_mean.clamp_min(1)
                ).tolist(),
            }
        summaries.append({"file": path.name, "phases": phases})
    if not summaries:
        raise AssertionError("expert recorder produced no files")
    return {
        "rank_records": summaries,
        "scope": "routed expert token counts under trivial fixed placement; replicated rank routing histograms are not summed; counts are not GPU time",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--ep-size", type=int, choices=[1, 2], required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    random.seed(42)
    report = {"seed": 42, "status": "started", "ep_size": args.ep_size}
    try:
        report["manifest"] = model_manifest(args.model)
        report["versions"] = {
            name: importlib.metadata.version(name)
            for name in [
                "sglang",
                "torch",
                "transformers",
                "triton",
                "flashinfer-python",
            ]
        }
        report["gpu_guard"] = guard_gpus()
        os.chdir(args.output)
        report["result"] = check_engine(
            args.model, args.output, args.record, args.ep_size
        )
        if args.record:
            report["routing"] = summarize_recorder(args.output, args.ep_size)
        report["status"] = "passed"
    except BaseException as error:
        report.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        (args.output / "status.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n"
        )
        print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
