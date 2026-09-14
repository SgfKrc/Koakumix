"""Live three-round ablation runner for ``EX-HARNESS-01``.

This module reuses exactly the arms, tasks, and metric fields of the offline
contract in :mod:`harness_workbench.research.harness_ablation`, but executes
the assembled context against a local ``llama-server`` (OpenAI-compatible
``/v1/chat/completions``) bound to the loopback interface.  Answers are judged
with ``loose_contains``; only their SHA-256, character count, and verdict are
recorded, never the answer text.

Scope: local inference only.  Non-loopback endpoints are rejected, no external
network is used, and the report keeps ``network_used=false`` while
``weights_loaded=true``.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence
from urllib import error as url_error
from urllib import parse as url_parse
from urllib import request as url_request

from ..context_engine.tokenizer import HeuristicTokenizer, TokenCounter
from .harness_ablation import (
    ARMS,
    DEFAULT_INPUT_BUDGET,
    HarnessAblationFixture,
    assemble_arm_context,
    build_harness_ablation_fixture,
)


SCHEMA = "qlh.harness.ablation.live.v1"
JUDGE_MODE = "loose_contains"
DEFAULT_ENDPOINT = "http://127.0.0.1:8081"
DEFAULT_MODEL_ID = "Qwen3-4B"
DEFAULT_ROUNDS = 3
DEFAULT_MAX_NEW_TOKENS = 96
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT_S = 120.0
ANSWER_CHAR_LIMIT = 2_000
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
JUDGE_PHRASES: Mapping[str, str] = {
    "over-budget-early-fact": "indigo lantern",
    "over-budget-constraint": "internal build numbers",
    "cross-session-units": "metric",
    "rag-budget-omissions": "silent truncation",
}
SYSTEM_PROMPT = (
    "You are a careful assistant. Answer the final question using only the "
    "supplied context. Reply in one short sentence."
)
CLOSING_INSTRUCTION = "Answer the last question above in one short sentence."

ChatTransport = Callable[[str, Mapping[str, Any], float], Mapping[str, Any]]
_NORMALIZE = re.compile(r"[\s\-_]+")


class LocalBackendError(RuntimeError):
    """Stable failure for loopback llama-server calls without payload leakage."""

    def __init__(self, code: str, status: int = 0) -> None:
        super().__init__(code)
        self.code = code
        self.status = status


def normalize_answer(text: str) -> str:
    """Lowercase and collapse whitespace, hyphens, and underscores."""

    return _NORMALIZE.sub(" ", text.lower()).strip()


def judge_answer(answer: str, task_id: str) -> bool:
    """Loose containment judge; the phrase must survive normalization."""

    phrase = JUDGE_PHRASES.get(task_id)
    if phrase is None:
        raise ValueError("unknown ablation task id")
    return normalize_answer(phrase) in normalize_answer(answer)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def assert_loopback_endpoint(endpoint: str) -> str:
    """Reject anything that is not a plain loopback HTTP endpoint."""

    parsed = url_parse.urlsplit(endpoint)
    if parsed.scheme != "http":
        raise ValueError("endpoint must use http on the loopback interface")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("endpoint must not carry credentials, query, or fragment")
    host = (parsed.hostname or "").lower()
    if host not in LOOPBACK_HOSTS:
        raise ValueError("endpoint host must be loopback")
    return endpoint.rstrip("/")


class UrllibChatTransport:
    """Minimal standard-library transport for the loopback chat endpoint."""

    def __call__(self, endpoint: str, payload: Mapping[str, Any], timeout: float) -> Mapping[str, Any]:
        url = assert_loopback_endpoint(endpoint) + "/v1/chat/completions"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = url_request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with url_request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
        except url_error.HTTPError as exc:
            raise LocalBackendError("backend_http_error", int(exc.code)) from None
        except (url_error.URLError, TimeoutError, OSError):
            raise LocalBackendError("backend_unreachable") from None
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise LocalBackendError("backend_invalid_json") from None
        if not isinstance(decoded, dict):
            raise LocalBackendError("backend_invalid_payload")
        return decoded


def extract_answer(payload: Mapping[str, Any]) -> tuple[str, int, int]:
    """Return ``(answer, prompt_tokens, completion_tokens)`` or a stable error."""

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LocalBackendError("backend_empty_choices")
    first = choices[0]
    message = first.get("message") if isinstance(first, Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, str) or not content.strip():
        raise LocalBackendError("backend_invalid_message")
    usage = payload.get("usage")
    usage = usage if isinstance(usage, Mapping) else {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    return (
        content.strip()[:ANSWER_CHAR_LIMIT],
        int(prompt_tokens) if isinstance(prompt_tokens, (int, float)) else 0,
        int(completion_tokens) if isinstance(completion_tokens, (int, float)) else 0,
    )


@dataclass(frozen=True, slots=True)
class LiveAblationObservation:
    """One live arm/task/round cell; answers are represented by their digest."""

    arm: str
    task_id: str
    task_kind: str
    round_index: int
    judge_phrase: str
    judge_mode: str
    task_success: bool
    answer_sha256: str
    answer_chars: int
    input_tokens: int
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    fixture_task_success: bool
    memory_recall_rate: float
    citation_expected: bool
    citation_correct: bool
    model_id: str
    endpoint_kind: str = "loopback"
    runner_kind: str = "llama_server"
    weights_loaded: bool = True
    network_used: bool = False
    notices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError("unsupported ablation arm")
        if self.judge_mode != JUDGE_MODE:
            raise ValueError("unsupported judge mode")
        if not re.fullmatch(r"[0-9a-f]{64}", self.answer_sha256):
            raise ValueError("answer digest must be a lowercase sha256")
        if self.answer_chars < 0 or self.input_tokens < 0 or self.round_index < 0:
            raise ValueError("cell counts must be non-negative")
        if self.prompt_tokens < 0 or self.completion_tokens < 0 or self.latency_ms < 0:
            raise ValueError("backend counters must be non-negative")
        if not 0.0 <= self.memory_recall_rate <= 1.0:
            raise ValueError("memory recall rate must be between zero and one")
        if not self.citation_expected and self.citation_correct:
            raise ValueError("citation correctness requires an expected citation")
        if self.endpoint_kind != "loopback" or self.runner_kind != "llama_server":
            raise ValueError("live observations must target a loopback llama-server")
        if not self.weights_loaded or self.network_used:
            raise ValueError("live observations load weights and must not claim external network use")

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "task_id": self.task_id,
            "task_kind": self.task_kind,
            "round_index": self.round_index,
            "judge_phrase": self.judge_phrase,
            "judge_mode": self.judge_mode,
            "task_success": self.task_success,
            "answer_sha256": self.answer_sha256,
            "answer_chars": self.answer_chars,
            "input_tokens": self.input_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": self.latency_ms,
            "fixture_task_success": self.fixture_task_success,
            "memory_recall_rate": self.memory_recall_rate,
            "citation_expected": self.citation_expected,
            "citation_correct": self.citation_correct,
            "model_id": self.model_id,
            "endpoint_kind": self.endpoint_kind,
            "runner_kind": self.runner_kind,
            "weights_loaded": self.weights_loaded,
            "network_used": self.network_used,
            "notices": list(self.notices),
        }


@dataclass(frozen=True, slots=True)
class LiveAblationArmSummary:
    """Aggregated live metrics per arm; latency stays outside the digest."""

    arm: str
    rounds: int
    task_count: int
    observation_count: int
    task_success_rate: float
    fixture_task_success_rate: float
    success_rate_margin: float
    stable: bool
    input_tokens_mean: float
    prompt_tokens_mean: float
    completion_tokens_mean: float
    token_cost_ratio: float
    latency_ms_mean: float
    latency_ms_min: int
    latency_ms_max: int

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError("unsupported ablation arm")
        if self.observation_count != self.task_count * self.rounds or self.rounds < DEFAULT_ROUNDS:
            raise ValueError("summary counts must match tasks times rounds")
        for value in (self.task_success_rate, self.fixture_task_success_rate, self.success_rate_margin):
            if not 0.0 <= value <= 1.0:
                raise ValueError("success rates must be between zero and one")
        if self.token_cost_ratio < 0 or self.latency_ms_mean < 0:
            raise ValueError("cost ratio and latency must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "rounds": self.rounds,
            "task_count": self.task_count,
            "observation_count": self.observation_count,
            "task_success_rate": self.task_success_rate,
            "fixture_task_success_rate": self.fixture_task_success_rate,
            "success_rate_margin": self.success_rate_margin,
            "stable": self.stable,
            "input_tokens_mean": self.input_tokens_mean,
            "prompt_tokens_mean": self.prompt_tokens_mean,
            "completion_tokens_mean": self.completion_tokens_mean,
            "token_cost_ratio": self.token_cost_ratio,
            "latency_ms_mean": self.latency_ms_mean,
            "latency_ms_min": self.latency_ms_min,
            "latency_ms_max": self.latency_ms_max,
        }


@dataclass(frozen=True, slots=True)
class LiveAblationReport:
    """Live three-arm report with a verdict-only, latency-free digest."""

    fixture: HarnessAblationFixture
    observations: tuple[LiveAblationObservation, ...]
    summaries: tuple[LiveAblationArmSummary, ...]
    endpoint_kind: str = "loopback"
    model_id: str = DEFAULT_MODEL_ID
    rounds: int = DEFAULT_ROUNDS
    input_budget: int = DEFAULT_INPUT_BUDGET
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    judge_mode: str = JUDGE_MODE
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError("unsupported live ablation schema")
        expected = len(ARMS) * len(self.fixture.tasks) * self.rounds
        if len(self.observations) != expected:
            raise ValueError("one observation is required for every arm/task/round cell")
        cells = {(item.arm, item.task_id, item.round_index) for item in self.observations}
        if len(cells) != expected:
            raise ValueError("observations must contain one unique cell per arm, task, and round")
        if {item.arm for item in self.summaries} != set(ARMS):
            raise ValueError("every arm requires a summary")
        if self.judge_mode != JUDGE_MODE or self.endpoint_kind != "loopback":
            raise ValueError("live reports are loopback-only with a single judge mode")

    def digest_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "fixture_digest": self.fixture.digest,
            "model_id": self.model_id,
            "rounds": self.rounds,
            "input_budget": self.input_budget,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "judge_mode": self.judge_mode,
            "cells": [
                {
                    "arm": item.arm,
                    "task_id": item.task_id,
                    "round_index": item.round_index,
                    "task_success": item.task_success,
                    "fixture_task_success": item.fixture_task_success,
                    "prompt_tokens": item.prompt_tokens,
                    "completion_tokens": item.completion_tokens,
                }
                for item in self.observations
            ],
            "summaries": [
                {
                    "arm": summary.arm,
                    "task_success_rate": summary.task_success_rate,
                    "fixture_task_success_rate": summary.fixture_task_success_rate,
                    "success_rate_margin": summary.success_rate_margin,
                    "stable": summary.stable,
                }
                for summary in self.summaries
            ],
        }

    @property
    def digest(self) -> str:
        return _digest(self.digest_payload())

    def series(self) -> dict[str, tuple[dict[str, Any], ...]]:
        return {
            arm: tuple(
                {
                    "round_index": item.round_index,
                    "task_id": item.task_id,
                    "task_success": item.task_success,
                    "fixture_task_success": item.fixture_task_success,
                    "latency_ms": item.latency_ms,
                    "completion_tokens": item.completion_tokens,
                }
                for item in self.observations
                if item.arm == arm
            )
            for arm in ARMS
        }

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "fixture": self.fixture.as_dict(),
            "config": {
                "endpoint_kind": self.endpoint_kind,
                "model_id": self.model_id,
                "rounds": self.rounds,
                "input_budget": self.input_budget,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "judge_mode": self.judge_mode,
            },
            "observations": [item.as_dict() for item in self.observations],
            "summaries": [summary.as_dict() for summary in self.summaries],
            "series": {key: list(items) for key, items in self.series().items()},
            "runner_kind": "llama_server",
            "weights_loaded": True,
            "network_used": False,
            "evidence_scope": "live_loopback_inference_with_loose_contains_judge",
        }
        if include_digest:
            value["report_digest"] = self.digest
        return value

    def to_markdown(self) -> str:
        lines = [
            "# EX-HARNESS-01 live ablation (loopback llama-server)",
            "",
            f"- Model: `{self.model_id}`; rounds: {self.rounds}; input budget: {self.input_budget}",
            f"- Judge: `{self.judge_mode}`; max new tokens: {self.max_new_tokens}; temperature: {self.temperature}",
            "- Runner: `llama_server` on loopback; weights loaded: `true`; external network used: `false`",
            "- Answers are stored as SHA-256 only; latency is excluded from the report digest.",
            "",
            "| arm | task success | fixture success | margin | stable | input tokens | prompt tokens | completion tokens | cost ratio | latency (ms mean/min/max) |",
            "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for summary in self.summaries:
            lines.append(
                f"| `{summary.arm}` | {summary.task_success_rate:.3f} | "
                f"{summary.fixture_task_success_rate:.3f} | {summary.success_rate_margin:.3f} | "
                f"{'yes' if summary.stable else 'no'} | {summary.input_tokens_mean:.1f} | "
                f"{summary.prompt_tokens_mean:.1f} | {summary.completion_tokens_mean:.1f} | "
                f"{summary.token_cost_ratio:.3f} | {summary.latency_ms_mean:.0f}/"
                f"{summary.latency_ms_min}/{summary.latency_ms_max} |"
            )
        lines.extend(
            (
                "",
                "## Per-task verdicts",
                "",
                "| arm | task | kind | verdict | judge phrase |",
                "| --- | --- | --- | --- | --- |",
            )
        )
        seen: set[tuple[str, str]] = set()
        for item in self.observations:
            key = (item.arm, item.task_id)
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                f"| `{item.arm}` | `{item.task_id}` | {item.task_kind} | "
                f"{'pass' if item.task_success else 'miss'} | `{item.judge_phrase}` |"
            )
        lines.extend(
            (
                "",
                f"Report digest: `{self.digest}`",
                "",
            )
        )
        return "\n".join(lines)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _summarize(
    arm: str,
    observations: tuple[LiveAblationObservation, ...],
    *,
    baseline_tokens_mean: float,
) -> LiveAblationArmSummary:
    cells = tuple(item for item in observations if item.arm == arm)
    rounds = sorted({item.round_index for item in cells})
    task_ids = sorted({item.task_id for item in cells})
    per_round = [
        _mean([1.0 if item.task_success else 0.0 for item in cells if item.round_index == round_index])
        for round_index in rounds
    ]
    margin = (max(per_round) - min(per_round)) if per_round else 0.0
    tokens_mean = _mean([float(item.input_tokens) for item in cells])
    latencies = [item.latency_ms for item in cells]
    return LiveAblationArmSummary(
        arm=arm,
        rounds=len(rounds),
        task_count=len(task_ids),
        observation_count=len(cells),
        task_success_rate=_mean([1.0 if item.task_success else 0.0 for item in cells]),
        fixture_task_success_rate=_mean([1.0 if item.fixture_task_success else 0.0 for item in cells]),
        success_rate_margin=round(margin, 9),
        stable=margin == 0.0,
        input_tokens_mean=round(tokens_mean, 3),
        prompt_tokens_mean=round(_mean([float(item.prompt_tokens) for item in cells]), 3),
        completion_tokens_mean=round(_mean([float(item.completion_tokens) for item in cells]), 3),
        token_cost_ratio=round(tokens_mean / baseline_tokens_mean, 6) if baseline_tokens_mean else 0.0,
        latency_ms_mean=round(_mean([float(value) for value in latencies]), 1),
        latency_ms_min=min(latencies) if latencies else 0,
        latency_ms_max=max(latencies) if latencies else 0,
    )


def run_live_ablation(
    fixture: HarnessAblationFixture | None = None,
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    model_id: str = DEFAULT_MODEL_ID,
    rounds: int = DEFAULT_ROUNDS,
    input_budget: int = DEFAULT_INPUT_BUDGET,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    transport: ChatTransport | None = None,
    tokenizer: TokenCounter | None = None,
) -> LiveAblationReport:
    """Run every arm/task/round cell against the loopback llama-server."""

    fixture = fixture or build_harness_ablation_fixture()
    if rounds < DEFAULT_ROUNDS:
        raise ValueError("live ablation needs at least three rounds")
    if input_budget <= 0 or max_new_tokens <= 0 or timeout_s <= 0:
        raise ValueError("budget, token limit, and timeout must be positive")
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("temperature must be between zero and two")
    normalized_endpoint = assert_loopback_endpoint(endpoint)
    caller = transport or UrllibChatTransport()
    counter = tokenizer or HeuristicTokenizer()
    observations: list[LiveAblationObservation] = []
    for arm in ARMS:
        for task in fixture.tasks:
            assembly = assemble_arm_context(task, arm, input_budget, tokenizer=counter)
            payload = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"{assembly.assembled_text}\n\n{CLOSING_INSTRUCTION}",
                    },
                ],
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            for round_index in range(rounds):
                started = time.perf_counter()
                response = caller(normalized_endpoint, payload, timeout_s)
                latency_ms = int((time.perf_counter() - started) * 1_000)
                answer, prompt_tokens, completion_tokens = extract_answer(response)
                observations.append(
                    LiveAblationObservation(
                        arm=arm,
                        task_id=task.task_id,
                        task_kind=task.kind,
                        round_index=round_index,
                        judge_phrase=JUDGE_PHRASES[task.task_id],
                        judge_mode=JUDGE_MODE,
                        task_success=judge_answer(answer, task.task_id),
                        answer_sha256=hashlib.sha256(answer.encode("utf-8")).hexdigest(),
                        answer_chars=len(answer),
                        input_tokens=assembly.input_tokens,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        latency_ms=latency_ms,
                        fixture_task_success=assembly.fixture_task_success,
                        memory_recall_rate=assembly.memory_recall_rate,
                        citation_expected=assembly.citation_expected,
                        citation_correct=assembly.citation_correct,
                        model_id=model_id,
                        notices=assembly.notices,
                    )
                )
    cells = tuple(observations)
    baseline_mean = _mean([float(item.input_tokens) for item in cells if item.arm == "baseline"])
    summaries = tuple(_summarize(arm, cells, baseline_tokens_mean=baseline_mean) for arm in ARMS)
    return LiveAblationReport(
        fixture=fixture,
        observations=cells,
        summaries=summaries,
        model_id=model_id,
        rounds=rounds,
        input_budget=input_budget,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


__all__ = [
    "ANSWER_CHAR_LIMIT",
    "CLOSING_INSTRUCTION",
    "DEFAULT_ENDPOINT",
    "DEFAULT_MODEL_ID",
    "DEFAULT_ROUNDS",
    "JUDGE_MODE",
    "JUDGE_PHRASES",
    "LIVE_ABLATION_SCHEMA",
    "LOOPBACK_HOSTS",
    "LocalBackendError",
    "LiveAblationArmSummary",
    "LiveAblationObservation",
    "LiveAblationReport",
    "UrllibChatTransport",
    "assert_loopback_endpoint",
    "extract_answer",
    "judge_answer",
    "normalize_answer",
    "run_live_ablation",
]


LIVE_ABLATION_SCHEMA = SCHEMA
