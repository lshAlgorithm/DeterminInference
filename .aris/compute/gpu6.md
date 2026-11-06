### env: moe@29da0d08
how: reuse isolated /home/tegao/infra-token-replay/.venv; no shared environment modifications
tier: GPU2,3; RTX4090 24GiB each; TP2/EP2 BF16; seed42
spec: experimental/token_replay/env-moe.json
weights: /home/tegao/infra-token-replay/models/Qwen1.5-MoE-A2.7B-Chat
validated: run6 independent agent invocation passed forward, exact replay and followup cache; full308-request workload and both-rank recorder coverage also passed separately; current documented run7 awaits fresh execution after recorder study
gotcha: JIT needs existing conda bin/ninja on PATH. Render chat template to text then encode for flat token IDs. Recorder directory must be supplied explicitly.

Invocation on gpu6 (output must not exist):
```bash
CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=/home/tegao/infra-token-replay/python CUDA_HOME=/home/tegao/infra-token-replay/cuda-toolkit PATH=/home/tegao/infra-token-replay/.venv/bin:/home/tegao/infra-token-replay/cuda-toolkit/bin:/home/tegao/miniconda3/envs/verl_qwen35_sglang/bin:/usr/local/bin:/usr/bin:/bin OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 /home/tegao/infra-token-replay/.venv/bin/python /home/tegao/infra-token-replay/moe_probe.py --model /home/tegao/infra-token-replay/models/Qwen1.5-MoE-A2.7B-Chat --output /home/tegao/infra-token-replay/moe-results-ep2-run7 --ep-size 2 --record
```
The probe checks GPU2/3 are free immediately before engine startup. This is a feasibility witness, not performance or real-workload validation.

Public stochastic sampling uses the existing cuRAND headers via scoped CPATH and a project-specific FlashInfer cache. The exact invocation is recorded in experimental/token_replay/stochastic-witness-command.txt; the fresh agent kernel witness passed with seed42 and no installed-package changes.
