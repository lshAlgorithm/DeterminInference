"""The AIPerf ``profile`` argv for one AgentX point."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from infx.bench import env as inputs
from infx.bench.agentic import traces

REQUIRED = (
    "MODEL",
    "MODEL_PREFIX",
    "FRAMEWORK",
    "CONC",
    "DURATION",
    "AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS",
    "AIPERF_EXPERIMENTAL_FAST",
    "AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID",
    "AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING",
)

MIN_SCENARIO_DURATION_S = 900
FAST_DURATION_S = 1200
FAST_WARMUP_REQUESTS_PER_LANE = "1"


@dataclass(frozen=True)
class ReplayConfig:
    """Everything ``replay_argv`` needs, validated."""

    url: str
    model: str
    """Served name sent as ``--model``."""
    tokenizer: str
    """``TOKENIZER`` (a recipe's checkpoint path), else the Hugging Face id; a wire name is not
    necessarily a valid repo id."""
    concurrency: int
    duration: int
    warmup_requests_per_lane: str | None
    live_failed_request_threshold: str | None
    warmup_grace_period: str | None
    extra_inputs: tuple[str, ...]
    dynamo_session_timeout: str | None
    """Set only when Dynamo conversation-aware routing applies."""
    max_context_length: int | None
    server_metrics_urls: tuple[str, ...]
    artifact_dir: Path
    unsafe_override: bool
    loader: str
    dataset: str
    apply_chat_template: bool
    benchmark_grace_period: str | None

    token_trace: Path | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str], result_dir: Path) -> ReplayConfig:
        """Validate the replay inputs in ``env``; artifacts go below ``result_dir``."""
        values = inputs.require(*REQUIRED, env=env)
        duration = inputs.parse_positive_int("DURATION", values["DURATION"])
        warmup = None
        if values["AIPERF_EXPERIMENTAL_FAST"] == "1":
            duration, warmup = FAST_DURATION_S, FAST_WARMUP_REQUESTS_PER_LANE
        conv_aware = (
            values["FRAMEWORK"].startswith("dynamo-")
            and values["AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING"] != "0"
            and values["AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID"] != "true"
        )
        max_context = inputs.optional("AIPERF_MAX_CONTEXT_LENGTH", env)
        loader, dataset = traces.resolve(
            values["MODEL_PREFIX"], inputs.optional("WEKA_LOADER_OVERRIDE", env)
        )
        token_trace = inputs.optional("AGENTX_TOKEN_TRACE", env)
        mode = inputs.optional("AGENTX_REPLAY_MODE", env)
        if token_trace or mode:
            if not token_trace or mode not in {"generated", "recorded"}:
                raise inputs.InputError(
                    "AGENTX_TOKEN_TRACE and AGENTX_REPLAY_MODE=generated|recorded are required together"
                )
            inputs.require("AGENTX_HARNESS_WHEEL", "AGENTX_HARNESS_SHA256", env=env)
            if "sglang" not in values["FRAMEWORK"]:
                raise inputs.InputError("token experiments require the patched SGLang backend")
            if not Path(token_trace).exists():
                raise inputs.InputError(f"AGENTX_TOKEN_TRACE does not exist: {token_trace}")
            if env.get("AIPERF_APPLY_CHAT_TEMPLATE") == "true":
                raise inputs.InputError("recorded prompt IDs already include the chat template")
        extra_inputs = inputs.optional("AIPERF_EXTRA_INPUTS", env) or ""
        return cls(
            url=server_url(env),
            model=inputs.optional("SERVED_MODEL_NAME", env) or values["MODEL"],
            tokenizer=inputs.optional("TOKENIZER", env) or values["MODEL"],
            concurrency=inputs.parse_positive_int("CONC", values["CONC"]),
            duration=duration,
            warmup_requests_per_lane=warmup,
            live_failed_request_threshold=inputs.optional(
                "AIPERF_LIVE_FAILED_REQUEST_THRESHOLD", env
            ),
            warmup_grace_period=inputs.optional("AGENTIC_WARMUP_GRACE_PERIOD", env),
            extra_inputs=tuple(extra_inputs.split()),
            dynamo_session_timeout=(
                values["AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS"] if conv_aware else None
            ),
            max_context_length=(
                inputs.parse_positive_int("AIPERF_MAX_CONTEXT_LENGTH", max_context)
                if max_context
                else None
            ),
            server_metrics_urls=metrics_urls(env),
            artifact_dir=result_dir / "aiperf_artifacts",
            unsafe_override=duration < MIN_SCENARIO_DURATION_S,
            loader=loader,
            dataset=dataset,
            apply_chat_template=env.get("AIPERF_APPLY_CHAT_TEMPLATE") == "true",
            benchmark_grace_period=inputs.optional("AIPERF_BENCHMARK_GRACE_PERIOD", env),
            token_trace=Path(token_trace).absolute() if token_trace else None,
        )


def server_url(env: Mapping[str, str]) -> str:
    """srt-slurm's frontend, else ``AIPERF_SERVER_URL``, else ``http://localhost:$PORT``."""
    host = inputs.optional("SRT_FRONTEND_HOST", env)
    if host:
        return f"http://{host}:{inputs.positive_int('SRT_FRONTEND_PORT', env)}"
    return inputs.optional("AIPERF_SERVER_URL", env) or (
        f"http://localhost:{inputs.positive_int('PORT', env)}"
    )


def metrics_urls(env: Mapping[str, str], path: str = "/metrics") -> tuple[str, ...]:
    """Every Prometheus endpoint AIPerf scrapes; it keeps ``endpoint_url`` per series.

    ``path`` is the workers' Prometheus route when the endpoints come from ``SRT_*``.
    """
    urls = inputs.optional("AIPERF_SERVER_METRICS_URLS", env)
    if not urls and env.get("SRTCTL_FRONTEND_TYPE") != "dynamo":
        prefill = env.get("SRT_PREFILL_ENDPOINTS")
        endpoints = env.get("SRT_AGG_ENDPOINTS") or (
            (f"{prefill}," if prefill else "") + env.get("SRT_DECODE_ENDPOINTS", "")
        )
        endpoints = endpoints.removesuffix(",")
        if endpoints:
            urls = ",".join(f"http://{e}{path}" if e else "" for e in endpoints.split(","))
    if not urls:
        return ()
    parts = urls.split(",")
    if parts[-1] == "":
        parts.pop()
    if not parts or any(not url or any(c.isspace() for c in url) for url in parts):
        raise inputs.InputError(
            "AIPERF_SERVER_METRICS_URLS must be a comma-separated list of non-empty URLs, "
            f"got {urls!r}"
        )
    return tuple(parts)


def replay_argv(cfg: ReplayConfig, cli: str | Path) -> list[str]:
    """The ``agentx profile`` argv for ``cfg``; ``cli`` is the ``agentx`` executable."""
    argv = [str(cli), "profile", "--url", cfg.url]
    argv += ["--model", cfg.model, "--tokenizer", cfg.tokenizer]
    argv += ["--concurrency", str(cfg.concurrency), "--benchmark-duration", str(cfg.duration)]
    if cfg.live_failed_request_threshold is not None:
        argv += ["--failed-request-threshold", cfg.live_failed_request_threshold]
    if cfg.warmup_requests_per_lane is not None:
        argv += ["--warmup-requests-per-lane", cfg.warmup_requests_per_lane]
    if cfg.warmup_grace_period is not None:
        argv += ["--warmup-grace-period", cfg.warmup_grace_period]
    if cfg.extra_inputs:
        argv += ["--extra-inputs", *cfg.extra_inputs]
    if cfg.dynamo_session_timeout is not None:
        argv += ["--use-dynamo-conv-aware-routing"]
        argv += ["--dynamo-session-timeout-seconds", cfg.dynamo_session_timeout]
    argv.append("--tokenizer-trust-remote-code")
    if cfg.max_context_length is not None:
        argv += ["--max-context-length", str(cfg.max_context_length)]
    if cfg.server_metrics_urls:
        argv += ["--server-metrics", *cfg.server_metrics_urls]
    argv += ["--output-artifact-dir", str(cfg.artifact_dir)]
    if cfg.unsafe_override:
        argv.append("--unsafe-override")
    if cfg.token_trace is None:
        argv += ["--public-dataset", cfg.loader]
    else:
        argv += ["--input-file", str(cfg.token_trace), "--custom-dataset-type", "weka_trace"]
        argv += ["--endpoint-type", "completions", "--random-seed", "42"]
        if "--unsafe-override" not in argv:
            argv.append("--unsafe-override")
    if cfg.apply_chat_template:
        argv.append("--apply-chat-template")
    if cfg.benchmark_grace_period is not None:
        argv += ["--benchmark-grace-period", cfg.benchmark_grace_period]
    return argv
