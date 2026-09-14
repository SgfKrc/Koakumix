"""Speculative-decoding role profile for ``EX-SPEC-ROLE-01``.

This module measures the draft/verify role split on a loopback llama-server:
acceptance rate, tokens per round, decode throughput, and output agreement
against a single-model baseline.  Hypothesis ids follow the CACHE-05 argument
(``ASYM-01`` output-distribution agreement, ``ASYM-02`` cost gate:
``tokens_per_round > 1.5`` and RTT below the local decode budget).

Execution is local inference only: endpoints must be loopback, answers are
recorded as SHA-256, and timing fields stay outside the report digest so a
re-run with identical verdicts keeps an identical digest.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence
from urllib import error as url_error
from urllib import request as url_request

from .ablation_live import (
    LocalBackendError,
    UrllibChatTransport,
    assert_loopback_endpoint,
    extract_answer,
)


SCHEMA = "qlh.harness.role_spec.v1"
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT_S = 180.0
SPEC_COUNTERS: Mapping[str, str] = {
    "draft_tokens": "llamacpp:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "llamacpp:spec_decode_num_accepted_tokens_total",
    "drafts": "llamacpp:spec_decode_num_drafts_total",
}
ASYM_HYPOTHESES = ("ASYM-01", "ASYM-02")
TOKENS_PER_ROUND_FLOOR = 1.5
DEFAULT_PROMPTS: tuple[tuple[str, str], ...] = (
    ("short-fact", "Name the capital of France in one short sentence."),
    ("short-explain", "Explain in two sentences why KV caching speeds up decoding."),
    ("short-code", "Write a Python one-liner that reverses a string."),
    ("list-three", "List exactly three benefits of speculative decoding, one per line."),
    ("short-math", "Compute 17 * 23 and show the result in one sentence."),
)

ChatTransport = Callable[[str, Mapping[str, Any], float], Mapping[str, Any]]
MetricsReader = Callable[[str], str]
_COUNTER_LINE = re.compile(
    r"^([A-Za-z_:][A-Za-z0-9_:]*)\s+([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)$"
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def parse_prometheus_counters(text: str) -> dict[str, float]:
    """Parse `name value` counter lines, ignoring HELP/TYPE comments."""

    counters: dict[str, float] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _COUNTER_LINE.match(stripped)
        if match:
            counters[match.group(1)] = float(match.group(2))
    return counters


def spec_delta(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, int]:
    """Per-call speculative counters derived from two metrics snapshots."""

    return {
        key: max(0, int(after.get(name, 0.0) - before.get(name, 0.0)))
        for key, name in SPEC_COUNTERS.items()
    }


class UrllibMetricsReader:
    """Standard-library reader for the loopback prometheus endpoint."""

    def __init__(self, timeout_s: float = 10.0) -> None:
        self.timeout_s = timeout_s

    def __call__(self, endpoint: str) -> str:
        url = assert_loopback_endpoint(endpoint) + "/metrics"
        try:
            with url_request.urlopen(url, timeout=self.timeout_s) as response:
                return response.read().decode("utf-8", errors="replace")
        except (url_error.HTTPError, url_error.URLError, TimeoutError, OSError):
            raise LocalBackendError("metrics_unavailable") from None


def extract_timings(payload: Mapping[str, Any]) -> tuple[int, float]:
    """Return ``(predicted_tokens, predicted_ms)`` from llama-server timings."""

    timings = payload.get("timings")
    if not isinstance(timings, Mapping):
        return (0, 0.0)
    predicted_n = timings.get("predicted_n")
    predicted_ms = timings.get("predicted_ms")
    tokens = int(predicted_n) if isinstance(predicted_n, (int, float)) else 0
    elapsed = float(predicted_ms) if isinstance(predicted_ms, (int, float)) else 0.0
    return (max(0, tokens), max(0.0, elapsed))


@dataclass(frozen=True, slots=True)
class SpecProfileConfig:
    """One server configuration under test (baseline or draft-verify)."""

    label: str
    target_model: str
    draft_model: str | None = None
    speculative: bool = False
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE

    def __post_init__(self) -> None:
        if not self.label or not re.fullmatch(r"[A-Za-z0-9_.:-]+", self.label):
            raise ValueError("profile label is invalid")
        if not self.target_model.strip():
            raise ValueError("target model is required")
        if self.speculative and not (self.draft_model or "").strip():
            raise ValueError("speculative profiles require a draft model")
        if not self.speculative and self.draft_model:
            raise ValueError("baseline profiles must not declare a draft model")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between zero and two")

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "target_model": self.target_model,
            "draft_model": self.draft_model,
            "speculative": self.speculative,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }


@dataclass(frozen=True, slots=True)
class SpecPromptResult:
    """One prompt executed under one profile."""

    config_label: str
    prompt_id: str
    answer_sha256: str
    answer_chars: int
    prompt_tokens: int
    completion_tokens: int
    predicted_tokens: int
    predicted_ms: float
    wall_ms: int
    decode_tokens_per_s: float
    draft_tokens: int
    accepted_tokens: int
    drafts: int
    speculative: bool

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.answer_sha256):
            raise ValueError("answer digest must be a lowercase sha256")
        counts = (
            self.answer_chars,
            self.prompt_tokens,
            self.completion_tokens,
            self.predicted_tokens,
            self.draft_tokens,
            self.accepted_tokens,
            self.drafts,
            self.wall_ms,
        )
        if any(value < 0 for value in counts):
            raise ValueError("result counts must be non-negative")
        if self.predicted_ms < 0 or self.decode_tokens_per_s < 0:
            raise ValueError("timings must be non-negative")
        if not self.speculative and (self.draft_tokens or self.accepted_tokens or self.drafts):
            raise ValueError("baseline profiles must not report speculative counters")
        if self.accepted_tokens > self.draft_tokens:
            raise ValueError("accepted tokens cannot exceed drafted tokens")

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_tokens / self.draft_tokens if self.draft_tokens else 0.0

    @property
    def tokens_per_round(self) -> float:
        return self.draft_tokens / self.drafts if self.drafts else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_label": self.config_label,
            "prompt_id": self.prompt_id,
            "answer_sha256": self.answer_sha256,
            "answer_chars": self.answer_chars,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "predicted_tokens": self.predicted_tokens,
            "predicted_ms": round(self.predicted_ms, 3),
            "wall_ms": self.wall_ms,
            "decode_tokens_per_s": round(self.decode_tokens_per_s, 3),
            "draft_tokens": self.draft_tokens,
            "accepted_tokens": self.accepted_tokens,
            "drafts": self.drafts,
            "acceptance_rate": round(self.acceptance_rate, 6),
            "tokens_per_round": round(self.tokens_per_round, 6),
            "speculative": self.speculative,
        }


@dataclass(frozen=True, slots=True)
class SpecProfileSummary:
    """Aggregated throughput, acceptance, and agreement for one profile."""

    label: str
    target_model: str
    draft_model: str | None
    speculative: bool
    prompt_count: int
    decode_tokens_per_s_mean: float
    wall_ms_mean: float
    acceptance_rate: float
    tokens_per_round: float
    speedup_vs_baseline: float
    agreement_with_baseline: float
    checks: Mapping[str, bool]
    baseline_label: str | None = None

    def __post_init__(self) -> None:
        if self.prompt_count <= 0:
            raise ValueError("summary needs at least one prompt")
        if self.decode_tokens_per_s_mean < 0 or self.wall_ms_mean < 0:
            raise ValueError("throughput and latency must be non-negative")
        for value in (self.acceptance_rate, self.agreement_with_baseline):
            if not 0.0 <= value <= 1.0:
                raise ValueError("rates must be between zero and one")
        if self.tokens_per_round < 0 or self.speedup_vs_baseline < 0:
            raise ValueError("tokens per round and speedup must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "target_model": self.target_model,
            "draft_model": self.draft_model,
            "speculative": self.speculative,
            "prompt_count": self.prompt_count,
            "decode_tokens_per_s_mean": round(self.decode_tokens_per_s_mean, 3),
            "wall_ms_mean": round(self.wall_ms_mean, 1),
            "acceptance_rate": round(self.acceptance_rate, 6),
            "tokens_per_round": round(self.tokens_per_round, 6),
            "speedup_vs_baseline": round(self.speedup_vs_baseline, 6),
            "agreement_with_baseline": round(self.agreement_with_baseline, 6),
            "baseline_label": self.baseline_label,
            "checks": dict(sorted(self.checks.items())),
        }


@dataclass(frozen=True, slots=True)
class SpecRoleReport:
    """Role-split report: configs, per-prompt results, summaries, and verdicts."""

    configs: tuple[SpecProfileConfig, ...]
    results: tuple[SpecPromptResult, ...]
    summaries: tuple[SpecProfileSummary, ...]
    baseline_label: str
    endpoint_kind: str = "loopback"
    judge_mode: str = "answer_sha256_agreement"
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError("unsupported role spec schema")
        if not self.configs or not self.results or not self.summaries:
            raise ValueError("role spec report needs configs, results, and summaries")
        labels = [config.label for config in self.configs]
        if len(set(labels)) != len(labels):
            raise ValueError("profile labels must be unique")
        if self.baseline_label not in set(labels):
            raise ValueError("baseline label must reference a configured profile")
        if {summary.label for summary in self.summaries} != set(labels):
            raise ValueError("every profile requires a summary")
        if self.endpoint_kind != "loopback":
            raise ValueError("role spec profiles are loopback-only")

    def digest_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "baseline_label": self.baseline_label,
            "configs": [config.as_dict() for config in self.configs],
            "results": [
                {
                    "config_label": item.config_label,
                    "prompt_id": item.prompt_id,
                    "answer_sha256": item.answer_sha256,
                    "prompt_tokens": item.prompt_tokens,
                    "completion_tokens": item.completion_tokens,
                    "predicted_tokens": item.predicted_tokens,
                    "draft_tokens": item.draft_tokens,
                    "accepted_tokens": item.accepted_tokens,
                    "drafts": item.drafts,
                }
                for item in self.results
            ],
            "summaries": [
                {
                    "label": summary.label,
                    "speculative": summary.speculative,
                    "acceptance_rate": summary.acceptance_rate,
                    "tokens_per_round": summary.tokens_per_round,
                    "agreement_with_baseline": summary.agreement_with_baseline,
                    "checks": dict(sorted(summary.checks.items())),
                }
                for summary in self.summaries
            ],
        }

    @property
    def digest(self) -> str:
        return _digest(self.digest_payload())

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "baseline_label": self.baseline_label,
            "endpoint_kind": self.endpoint_kind,
            "judge_mode": self.judge_mode,
            "hypotheses": list(ASYM_HYPOTHESES),
            "configs": [config.as_dict() for config in self.configs],
            "results": [item.as_dict() for item in self.results],
            "summaries": [summary.as_dict() for summary in self.summaries],
            "runner_kind": "llama_server",
            "weights_loaded": True,
            "network_used": False,
            "evidence_scope": "loopback_speculative_role_profile_with_timing_excluded_digest",
        }
        if include_digest:
            value["report_digest"] = self.digest
        return value

    def to_markdown(self) -> str:
        lines = [
            "# EX-SPEC-ROLE-01 speculative role profile",
            "",
            "- Runner: `llama_server` on loopback; weights loaded: `true`; external network used: `false`",
            f"- Baseline profile: `{self.baseline_label}`; hypotheses: {', '.join(ASYM_HYPOTHESES)}",
            "- Acceptance rate = accepted / drafted tokens; tokens/round = drafted / verification steps.",
            "- Timing fields are excluded from the report digest.",
            "",
            "| profile | target | draft | speculative | decode tok/s | acceptance | tokens/round | speedup | agreement | ASYM-01 | ASYM-02 |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
        for summary in self.summaries:
            checks = summary.checks
            lines.append(
                f"| `{summary.label}` | `{summary.target_model}` | "
                f"{'`' + summary.draft_model + '`' if summary.draft_model else '-'} | "
                f"{'yes' if summary.speculative else 'no'} | {summary.decode_tokens_per_s_mean:.2f} | "
                f"{summary.acceptance_rate:.3f} | {summary.tokens_per_round:.2f} | "
                f"{summary.speedup_vs_baseline:.3f} | {summary.agreement_with_baseline:.3f} | "
                f"{'pass' if checks.get('asym_01_distribution_agreement') else 'fail'} | "
                f"{'pass' if checks.get('asym_02_cost_gate') else 'fail'} |"
            )
        lines.extend(
            (
                "",
                "## Per-prompt results",
                "",
                "| profile | prompt | completion tokens | decode tok/s | drafted | accepted | rounds |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            )
        )
        for item in self.results:
            lines.append(
                f"| `{item.config_label}` | `{item.prompt_id}` | {item.completion_tokens} | "
                f"{item.decode_tokens_per_s:.2f} | {item.draft_tokens} | {item.accepted_tokens} | {item.drafts} |"
            )
        lines.extend(("", f"Report digest: `{self.digest}`", ""))
        return "\n".join(lines)


def _chat_payload(config: SpecProfileConfig, prompt: str) -> dict[str, Any]:
    return {
        "model": config.target_model,
        "messages": [
            {"role": "system", "content": "Answer directly and concisely."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def run_spec_profile(
    config: SpecProfileConfig,
    *,
    endpoint: str,
    prompts: Sequence[tuple[str, str]] = DEFAULT_PROMPTS,
    transport: ChatTransport | None = None,
    metrics_reader: MetricsReader | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> tuple[SpecPromptResult, ...]:
    """Execute every prompt once under one profile, with per-call spec counters."""

    if not prompts:
        raise ValueError("at least one prompt is required")
    normalized = assert_loopback_endpoint(endpoint)
    caller = transport or UrllibChatTransport()
    reader = metrics_reader or UrllibMetricsReader()
    results: list[SpecPromptResult] = []
    for prompt_id, prompt in prompts:
        before = parse_prometheus_counters(reader(normalized))
        started = time_perf_counter_ms()
        response = caller(normalized, _chat_payload(config, prompt), timeout_s)
        wall_ms = int(time_perf_counter_ms() - started)
        after = parse_prometheus_counters(reader(normalized))
        answer, prompt_tokens, completion_tokens = extract_answer(response)
        predicted_tokens, predicted_ms = extract_timings(response)
        delta = spec_delta(before, after)
        decode_per_s = (predicted_tokens / (predicted_ms / 1000.0)) if predicted_ms > 0 else 0.0
        results.append(
            SpecPromptResult(
                config_label=config.label,
                prompt_id=prompt_id,
                answer_sha256=hashlib.sha256(answer.encode("utf-8")).hexdigest(),
                answer_chars=len(answer),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                predicted_tokens=predicted_tokens or completion_tokens,
                predicted_ms=predicted_ms,
                wall_ms=wall_ms,
                decode_tokens_per_s=decode_per_s,
                draft_tokens=delta["draft_tokens"],
                accepted_tokens=delta["accepted_tokens"],
                drafts=delta["drafts"],
                speculative=config.speculative,
            )
        )
    return tuple(results)


def time_perf_counter_ms() -> float:
    """Monotonic millisecond clock."""

    return time.perf_counter() * 1000.0


def _summary(
    config: SpecProfileConfig,
    results: tuple[SpecPromptResult, ...],
    *,
    baseline: tuple[SpecPromptResult, ...] | None,
    baseline_label: str | None = None,
) -> SpecProfileSummary:
    cells = tuple(item for item in results if item.config_label == config.label)
    if not cells:
        raise ValueError("profile produced no results")
    decode_mean = sum(item.decode_tokens_per_s for item in cells) / len(cells)
    wall_mean = sum(item.wall_ms for item in cells) / len(cells)
    drafted = sum(item.draft_tokens for item in cells)
    accepted = sum(item.accepted_tokens for item in cells)
    rounds = sum(item.drafts for item in cells)
    acceptance_rate = accepted / drafted if drafted else 0.0
    tokens_per_round = drafted / rounds if rounds else 0.0
    agreement = 0.0
    if baseline is not None:
        by_prompt = {item.prompt_id: item.answer_sha256 for item in baseline}
        matches = [
            item for item in cells if by_prompt.get(item.prompt_id) == item.answer_sha256
        ]
        agreement = len(matches) / len(cells) if cells else 0.0
    baseline_decode = None
    if baseline is not None:
        baseline_decode = sum(item.decode_tokens_per_s for item in baseline) / len(baseline) if baseline else 0.0
    speedup = (decode_mean / baseline_decode) if baseline_decode else 1.0
    checks: dict[str, bool] = {
        "asym_01_distribution_agreement": agreement >= 1.0 if baseline is not None else True,
        "asym_02_cost_gate": (not config.speculative) or (tokens_per_round > TOKENS_PER_ROUND_FLOOR and speedup >= 1.0),
    }
    return SpecProfileSummary(
        label=config.label,
        target_model=config.target_model,
        draft_model=config.draft_model,
        speculative=config.speculative,
        prompt_count=len(cells),
        decode_tokens_per_s_mean=decode_mean,
        wall_ms_mean=wall_mean,
        acceptance_rate=acceptance_rate,
        tokens_per_round=tokens_per_round,
        speedup_vs_baseline=speedup,
        agreement_with_baseline=agreement,
        checks=checks,
        baseline_label=baseline_label,
    )


def select_baseline(
    config: SpecProfileConfig,
    configs: Sequence[SpecProfileConfig],
    results_by_label: Mapping[str, Sequence[SpecPromptResult]],
    *,
    fallback_label: str,
) -> tuple[str | None, tuple[SpecPromptResult, ...] | None]:
    """Pick the comparison baseline: same target first, then the global baseline.

    Comparing a 7B target against a 4B baseline would measure the target swap
    rather than the draft/verify split, so a same-target solo profile wins.
    """

    if not config.speculative:
        return (None, None)
    same_target = next(
        (
            item
            for item in configs
            if item.label != config.label
            and not item.speculative
            and item.target_model == config.target_model
        ),
        None,
    )
    label = same_target.label if same_target is not None else fallback_label
    if label == config.label:
        return (None, None)
    return (label, tuple(results_by_label[label]))


def _summarize_with_baseline(
    config: SpecProfileConfig,
    configs: Sequence[SpecProfileConfig],
    results_by_label: Mapping[str, Sequence[SpecPromptResult]],
    *,
    fallback_label: str,
) -> SpecProfileSummary:
    label, baseline = select_baseline(
        config, configs, results_by_label, fallback_label=fallback_label
    )
    return _summary(
        config,
        tuple(results_by_label[config.label]),
        baseline=baseline,
        baseline_label=label,
    )


def summarize_profile(
    config: SpecProfileConfig,
    results: Sequence[SpecPromptResult],
    *,
    baseline: Sequence[SpecPromptResult] | None = None,
    baseline_label: str | None = None,
) -> SpecProfileSummary:
    """Public wrapper so callers can summarize results collected out of band."""

    return _summary(
        config,
        tuple(results),
        baseline=None if baseline is None else tuple(baseline),
        baseline_label=baseline_label,
    )


def build_spec_role_report(
    configs: Sequence[SpecProfileConfig],
    results_by_label: Mapping[str, Sequence[SpecPromptResult]],
    *,
    baseline_label: str,
) -> SpecRoleReport:
    """Assemble a report from results already collected per profile.

    This is the entry point for profiles that cannot share one server process:
    collect each profile, then build the comparison here.
    """

    if not configs:
        raise ValueError("at least one profile configuration is required")
    labels = [config.label for config in configs]
    if baseline_label not in labels:
        raise ValueError("baseline label must reference a configured profile")
    missing = [label for label in labels if label not in results_by_label]
    if missing:
        raise ValueError("missing results for configured profiles")
    summaries = tuple(
        _summarize_with_baseline(
            config, configs, results_by_label, fallback_label=baseline_label
        )
        for config in configs
    )
    results = tuple(item for config in configs for item in results_by_label[config.label])
    return SpecRoleReport(
        configs=tuple(configs),
        results=results,
        summaries=summaries,
        baseline_label=baseline_label,
    )


def run_spec_role_report(
    configs: Sequence[SpecProfileConfig],
    *,
    baseline_label: str,
    endpoint: str,
    prompts: Sequence[tuple[str, str]] = DEFAULT_PROMPTS,
    transport: ChatTransport | None = None,
    metrics_reader: MetricsReader | None = None,
    transport_factory: Callable[[SpecProfileConfig], ChatTransport] | None = None,
    metrics_reader_factory: Callable[[SpecProfileConfig], MetricsReader] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> SpecRoleReport:
    """Run every profile in order and compare each against its matched baseline.

    ``transport_factory`` / ``metrics_reader_factory`` let callers bind one
    transport per profile, which is what a per-configuration server needs.
    """

    if not configs:
        raise ValueError("at least one profile configuration is required")
    baseline_config = next((item for item in configs if item.label == baseline_label), None)
    if baseline_config is None:
        raise ValueError("baseline label must reference a configured profile")
    collected: dict[str, tuple[SpecPromptResult, ...]] = {}
    for config in configs:
        caller = transport_factory(config) if transport_factory is not None else transport
        reader = metrics_reader_factory(config) if metrics_reader_factory is not None else metrics_reader
        collected[config.label] = run_spec_profile(
            config,
            endpoint=endpoint,
            prompts=prompts,
            transport=caller,
            metrics_reader=reader,
            timeout_s=timeout_s,
        )
    summaries = tuple(
        _summarize_with_baseline(
            config, configs, collected, fallback_label=baseline_label
        )
        for config in configs
    )
    results = tuple(item for config in configs for item in collected[config.label])
    return SpecRoleReport(
        configs=tuple(configs),
        results=results,
        summaries=summaries,
        baseline_label=baseline_label,
    )


__all__ = [
    "ASYM_HYPOTHESES",
    "DEFAULT_PROMPTS",
    "SPEC_COUNTERS",
    "SPEC_ROLE_SCHEMA",
    "TOKENS_PER_ROUND_FLOOR",
    "LocalBackendError",
    "SpecProfileConfig",
    "SpecProfileSummary",
    "SpecPromptResult",
    "SpecRoleReport",
    "UrllibMetricsReader",
    "build_spec_role_report",
    "extract_timings",
    "parse_prometheus_counters",
    "run_spec_profile",
    "run_spec_role_report",
    "select_baseline",
    "spec_delta",
    "summarize_profile",
]


SPEC_ROLE_SCHEMA = SCHEMA
