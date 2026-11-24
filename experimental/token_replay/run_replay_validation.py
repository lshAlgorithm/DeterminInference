from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from run_campaign import run_point


def main(spec_path: Path, output: Path) -> None:
    frozen = json.loads(spec_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    (output / "spec.json").write_text(json.dumps(frozen, indent=2) + "\n")
    identity = {
        "files": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in frozen["identity_files"]
        },
        "compute_processes": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                "--format=csv,noheader",
            ],
            text=True,
        ),
        "pid": os.getpid(),
    }
    (output / "source-identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    state = {"status": "running", "pid": os.getpid(), "stages": []}
    try:
        for stage in frozen["stages"]:
            name = stage["name"]
            state["current"] = name
            (output / "status.json").write_text(json.dumps(state, indent=2) + "\n")
            server = {
                **frozen["server"],
                "env": {
                    **frozen["server"]["env"],
                    "TRITON_CACHE_DIR": frozen["cache_root"] + "/" + name + "/triton",
                    "TORCHINDUCTOR_CACHE_DIR": frozen["cache_root"]
                    + "/"
                    + name
                    + "/inductor",
                    "CUDA_CACHE_PATH": frozen["cache_root"] + "/" + name + "/cuda",
                },
            }
            for key in (
                "TRITON_CACHE_DIR",
                "TORCHINDUCTOR_CACHE_DIR",
                "CUDA_CACHE_PATH",
            ):
                Path(server["env"][key]).mkdir(parents=True, exist_ok=False)
            client = {
                **frozen["client"],
                "env": {
                    **frozen["client"]["env"],
                    "REPLAY_VALIDATION_SUITE": stage["suite"],
                },
            }
            spec = {**frozen, "server": server, "client": client}
            point = {
                "name": name,
                "mode": "recorded",
                "trace": "",
                "capacity": 32768,
                "concurrency": stage["concurrency"],
                "duration": stage["duration"],
            }
            try:
                run_point(spec, point, output)
                result = json.loads((output / name / "result.json").read_text())
                state["stages"].append(
                    {"name": name, "execution": "complete", "passed": result["passed"]}
                )
            except Exception as error:
                state["stages"].append(
                    {"name": name, "execution": "failed", "error": repr(error)}
                )
            (output / "status.json").write_text(json.dumps(state, indent=2) + "\n")
        state["status"] = "complete"
        state["passed"] = all(stage.get("passed", False) for stage in state["stages"])
    finally:
        (output / "status.json").write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.spec, args.output)
