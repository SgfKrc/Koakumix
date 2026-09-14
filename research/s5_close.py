"""S5 closeout report for ``S5-CLOSE-01``.

Aggregates four auditable sections into one defense-ready closeout report:

1. **contract drift** — harness ``/v1`` vs main-project ``/api`` semantics
   (reuses the offline API workbench report);
2. **Ollama comparison** — Ollama vs the loopback llama-server on the same
   prompt set; ``not_run`` unless transports are injected, never faked;
3. **Pareto** — quality/latency/RSS marking via the ceiling-study contract;
   fixture rows stay ``scope=fixture`` and carry a non-promotable boundary;
4. **evidence ledger** — ticket-level numbers, evidence paths, scope, and an
   explicit "what this does not prove" boundary per ticket.

The default run is offline.  Latency stays outside the report digest so a
re-run with identical verdicts keeps an identical digest.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .ablation_live import LocalBackendError, UrllibChatTransport, assert_loopback_endpoint, extract_answer
from .ceiling import pareto_points
from ..tools.api_workbench import APIWorkbenchReport, run_api_workbench


SCHEMA = "qlh.harness.s5_close.v1"
SECTIONS = ("contract_drift", "ollama_comparison", "pareto", "evidence_ledger")
EVIDENCE_SCOPES = ("fixture", "offline_contract", "live_loopback")
DEFAULT_OLLAMA_ENDPOINT = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen3-vl:4b"
DEFAULT_BASELINE_ENDPOINT = "http://127.0.0.1:8082"
DEFAULT_BASELINE_MODEL = "Qwen3-4B"
DEFAULT_CLOSE_PROMPTS: tuple[tuple[str, str], ...] = (
    ("short-fact", "Name the capital of France in one short sentence."),
    ("short-explain", "Explain in two sentences why prefix caching helps long chats."),
    ("list-three", "List exactly three benefits of retrieval augmented generation, one per line."),
)
DEFAULT_MAX_NEW_TOKENS = 96
DEFAULT_TEMPERATURE = 0.0

ChatTransport = Callable[[str, Mapping[str, Any], float], Mapping[str, Any]]


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def default_pareto_rows() -> tuple[dict[str, Any], ...]:
    """Fixture rows for the sub-1B role study; never a production selection."""

    return (
        {
            "id": "QW1.8B",
            "quality_rate": 0.250,
            "latency_p95_ms": 2450.0,
            "rss_peak_bytes": 2_400_000_000,
        },
        {
            "id": "Qwen3-0.6B",
            "quality_rate": 0.180,
            "latency_p95_ms": 900.0,
            "rss_peak_bytes": 1_500_000_000,
        },
        {
            "id": "Qwen2.5-0.5B",
            "quality_rate": 0.150,
            "latency_p95_ms": 820.0,
            "rss_peak_bytes": 1_300_000_000,
        },
        {
            "id": "MiniCPM4-0.5B",
            "quality_rate": 0.160,
            "latency_p95_ms": 760.0,
            "rss_peak_bytes": 1_250_000_000,
        },
    )


def default_evidence_ledger() -> tuple["EvidenceEntry", ...]:
    """Ticket-level evidence known as of 2026-09-15, with explicit boundaries."""

    return (
        EvidenceEntry(
            ticket="EX-HARNESS-01",
            theme="Koakumix 增强有效性对照实验（普通 chat / +RAG+记忆 / +压缩+角色）",
            status="completed",
            key_numbers="4B Q4_K_M：基线 0/4、+RAG+记忆 4/4、+压缩+角色 3/4；成本比 1.000/1.111/1.422；三轮波动 0；延迟 272–355 ms",
            evidence_paths=("build/ablation-ex-harness-01/ablation-report.md", "build/ablation-ex-harness-01/live/live-report.md"),
            scope="live_loopback",
            boundary="离线与真机判定一致，但任务集为 4 条固定 fixture；不构成通用质量结论",
        ),
        EvidenceEntry(
            ticket="EX-SPEC-ROLE-01",
            theme="投机解码角色实验（draft/verify 接受率与成本曲线）",
            status="completed",
            key_numbers="最佳 DS3-7B+Qwen2.5-0.5B = 1.342×（接受率 0.453）；4B+0.6B = 1.026×；两条启用路径 ASYM-01 均 fail（agreement 0.800）",
            evidence_paths=("build/role-spec-01/role-spec-report.md", "build/role-spec-01/vocab-mismatch-d7-qwen3draft.log"),
            scope="live_loopback",
            boundary="每配置 1 轮 ×5 prompt；词表族不匹配时投机被 llama.cpp 拒绝，接受率恒 0",
        ),
        EvidenceEntry(
            ticket="EX-CTX-MEAS-01",
            theme="30 轮上下文策略测度（滑窗 / STATE 摘要 / 长期记忆）",
            status="completed",
            key_numbers="6 预算 ×3 策略矩阵；滑窗超预算早期事实召回 0，STATE/记忆路径可召回",
            evidence_paths=("harness_workbench/research/context_measure.py",),
            scope="offline_contract",
            boundary="度量上下文保留，不是模型答案质量",
        ),
        EvidenceEntry(
            ticket="S3.2-RT-01/02",
            theme="红队 fixture 与 fail-closed gate（prompt injection / 工具越权 / 图片路径 / 上下文注入）",
            status="completed",
            key_numbers="12 条攻击样本全部 blocked；默认红队拦截率 100%、越权通过 0、decision schema 有效率 ≥98%",
            evidence_paths=("harness_workbench/eval/red_team.py",),
            scope="offline_contract",
            boundary="契约层拦截，不含真实生图模型的安全审计",
        ),
        EvidenceEntry(
            ticket="S8-MEM-01~03 + S8-E2E-01",
            theme="跨会话长期记忆（SQLite facts/preferences/decisions + FTS5 检索 + 压缩双写）",
            status="completed",
            key_numbers="scope 硬过滤、软删除保留审计行、fingerprint 去重、三层共享预算 omitted_count/truncated 可见",
            evidence_paths=("harness_workbench/memory/",),
            scope="offline_contract",
            boundary="不宣称记忆质量或生产 API 验收",
        ),
        EvidenceEntry(
            ticket="S7-NET-01/02 + S7-TOOL-01 + S7-MCP-01/02",
            theme="受限联网工具、工具结果回灌 gate、MCP 服务与 API bridge",
            status="completed",
            key_numbers="SSRF/DNS/重定向/大小/type 门全过；无 verified 能力时 fail-closed；MCP 工具目录含 chat/sessions/rag/memory/images/web",
            evidence_paths=("harness_workbench/tools/network.py", "harness_workbench/mcp_server/", "harness_workbench/api_layer/app.py"),
            scope="offline_contract",
            boundary="生产网络默认关闭；真实第三方 MCP 服务与认证仍后置",
        ),
        EvidenceEntry(
            ticket="S5-CLOSE-01",
            theme="S5 收口（契约漂移 + Ollama 对照 + Pareto + 演示证据汇总）",
            status="completed",
            key_numbers="本报告：契约漂移段离线可复现；Ollama 对照段仅在注入传输时运行；Pareto 段 fixture 行不参与生产选择",
            evidence_paths=("harness_workbench/research/s5_close.py",),
            scope="offline_contract",
            boundary="收口报告骨架；真机质量与公开演示数字仍需逐票证据支撑",
        ),
    )


@dataclass(frozen=True, slots=True)
class ContractDriftSection:
    """Harness-vs-main contract comparison carried from the API workbench."""

    valid: bool
    case_count: int
    matched: int
    drifted: int
    failed: int
    drift_cases: tuple[tuple[str, str], ...]
    runner_kind: str
    network_used: bool
    report_digest: str

    def __post_init__(self) -> None:
        if self.case_count <= 0 or min(self.matched, self.drifted, self.failed) < 0:
            raise ValueError("contract drift counts are invalid")
        if self.matched + self.drifted + self.failed != self.case_count:
            raise ValueError("contract drift counts must cover every case")
        if not re.fullmatch(r"[0-9a-f]{64}", self.report_digest):
            raise ValueError("contract drift digest must be a lowercase sha256")

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "case_count": self.case_count,
            "matched": self.matched,
            "drifted": self.drifted,
            "failed": self.failed,
            "drift_cases": [list(item) for item in self.drift_cases],
            "runner_kind": self.runner_kind,
            "network_used": self.network_used,
            "report_digest": self.report_digest,
        }


def build_contract_drift_section(report: APIWorkbenchReport | None = None) -> ContractDriftSection:
    """Summarize the API workbench report; default run is the offline fixture."""

    report = report or run_api_workbench()
    drift_cases = tuple(
        (item.case_id, "; ".join(item.mismatches) or "shape_or_status")
        for item in report.cases
        if item.status == "drifted"
    )
    summary = report.as_dict()["summary"]
    return ContractDriftSection(
        valid=report.valid,
        case_count=int(summary["case_count"]),
        matched=int(summary["matched"]),
        drifted=int(summary["drifted"]),
        failed=int(summary["failed"]),
        drift_cases=drift_cases,
        runner_kind=report.runner_kind,
        network_used=report.network_used,
        report_digest=report.digest,
    )


@dataclass(frozen=True, slots=True)
class OllamaComparisonEntry:
    """One prompt answered by both Ollama and the loopback llama-server."""

    prompt_id: str
    ollama_sha256: str
    ollama_tokens: int
    ollama_latency_ms: int
    baseline_sha256: str
    baseline_tokens: int
    baseline_latency_ms: int

    def __post_init__(self) -> None:
        if not self.prompt_id.strip():
            raise ValueError("prompt id is required")
        for value in (self.ollama_sha256, self.baseline_sha256):
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("answer digests must be lowercase sha256")
        if min(self.ollama_tokens, self.ollama_latency_ms, self.baseline_tokens, self.baseline_latency_ms) < 0:
            raise ValueError("comparison counters must be non-negative")

    @property
    def agreement(self) -> bool:
        return self.ollama_sha256 == self.baseline_sha256

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "ollama_sha256": self.ollama_sha256,
            "ollama_tokens": self.ollama_tokens,
            "ollama_latency_ms": self.ollama_latency_ms,
            "baseline_sha256": self.baseline_sha256,
            "baseline_tokens": self.baseline_tokens,
            "baseline_latency_ms": self.baseline_latency_ms,
            "agreement": self.agreement,
        }


@dataclass(frozen=True, slots=True)
class OllamaComparisonSection:
    """Ollama-vs-llama-server section; ``not_run`` until transports are given."""

    status: str
    ollama_model: str
    baseline_model: str
    entries: tuple[OllamaComparisonEntry, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"not_run", "completed"}:
            raise ValueError("unsupported comparison status")
        if self.status == "not_run" and self.entries:
            raise ValueError("not_run comparison must not carry entries")
        if self.status == "not_run" and not (self.reason or "").strip():
            raise ValueError("not_run comparison needs a reason")
        if self.status == "completed" and not self.entries:
            raise ValueError("completed comparison needs entries")

    @property
    def agreement_rate(self) -> float:
        if not self.entries:
            return 0.0
        return sum(1 for item in self.entries if item.agreement) / len(self.entries)

    @property
    def ollama_latency_mean(self) -> float:
        return sum(item.ollama_latency_ms for item in self.entries) / len(self.entries) if self.entries else 0.0

    @property
    def baseline_latency_mean(self) -> float:
        return sum(item.baseline_latency_ms for item in self.entries) / len(self.entries) if self.entries else 0.0

    @property
    def latency_ratio(self) -> float:
        baseline = self.baseline_latency_mean
        return self.ollama_latency_mean / baseline if baseline else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ollama_model": self.ollama_model,
            "baseline_model": self.baseline_model,
            "reason": self.reason,
            "prompt_count": len(self.entries),
            "agreement_rate": round(self.agreement_rate, 6),
            "ollama_latency_ms_mean": round(self.ollama_latency_mean, 1),
            "baseline_latency_ms_mean": round(self.baseline_latency_mean, 1),
            "latency_ratio": round(self.latency_ratio, 6),
            "entries": [item.as_dict() for item in self.entries],
        }


@dataclass(frozen=True, slots=True)
class OllamaComparisonConfig:
    """Endpoints, models, and prompts for the Ollama comparison section."""

    ollama_endpoint: str = DEFAULT_OLLAMA_ENDPOINT
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    baseline_endpoint: str = DEFAULT_BASELINE_ENDPOINT
    baseline_model: str = DEFAULT_BASELINE_MODEL
    prompts: tuple[tuple[str, str], ...] = DEFAULT_CLOSE_PROMPTS
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE

    def __post_init__(self) -> None:
        assert_loopback_endpoint(self.ollama_endpoint)
        assert_loopback_endpoint(self.baseline_endpoint)
        if not self.ollama_model.strip() or not self.baseline_model.strip():
            raise ValueError("both model names are required")
        if not self.prompts:
            raise ValueError("at least one prompt is required")
        if self.max_new_tokens <= 0 or not 0.0 <= self.temperature <= 2.0:
            raise ValueError("generation limits are invalid")


def _chat_payload(model: str, prompt: str, max_new_tokens: int, temperature: float) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "Answer directly and concisely."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def run_ollama_comparison(
    config: OllamaComparisonConfig | None = None,
    *,
    ollama_transport: ChatTransport | None = None,
    baseline_transport: ChatTransport | None = None,
    timeout_s: float = 180.0,
) -> OllamaComparisonSection:
    """Compare both endpoints; without transports the section stays ``not_run``."""

    config = config or OllamaComparisonConfig()
    if ollama_transport is None or baseline_transport is None:
        return OllamaComparisonSection(
            status="not_run",
            ollama_model=config.ollama_model,
            baseline_model=config.baseline_model,
            reason="transports_not_injected",
        )
    import time

    entries: list[OllamaComparisonEntry] = []
    for prompt_id, prompt in config.prompts:
        payload = _chat_payload(config.ollama_model, prompt, config.max_new_tokens, config.temperature)
        started = time.perf_counter()
        ollama_response = ollama_transport(config.ollama_endpoint, payload, timeout_s)
        ollama_latency = int((time.perf_counter() - started) * 1000)
        ollama_answer, _, ollama_tokens = extract_answer(ollama_response)

        payload = _chat_payload(config.baseline_model, prompt, config.max_new_tokens, config.temperature)
        started = time.perf_counter()
        baseline_response = baseline_transport(config.baseline_endpoint, payload, timeout_s)
        baseline_latency = int((time.perf_counter() - started) * 1000)
        baseline_answer, _, baseline_tokens = extract_answer(baseline_response)

        entries.append(
            OllamaComparisonEntry(
                prompt_id=prompt_id,
                ollama_sha256=hashlib.sha256(ollama_answer.encode("utf-8")).hexdigest(),
                ollama_tokens=ollama_tokens,
                ollama_latency_ms=ollama_latency,
                baseline_sha256=hashlib.sha256(baseline_answer.encode("utf-8")).hexdigest(),
                baseline_tokens=baseline_tokens,
                baseline_latency_ms=baseline_latency,
            )
        )
    return OllamaComparisonSection(
        status="completed",
        ollama_model=config.ollama_model,
        baseline_model=config.baseline_model,
        entries=tuple(entries),
    )


@dataclass(frozen=True, slots=True)
class ParetoSection:
    """Ceiling-study Pareto marking with an explicit scope label."""

    scope: str
    points: tuple[Mapping[str, Any], ...]
    dominated_count: int

    def __post_init__(self) -> None:
        if self.scope not in EVIDENCE_SCOPES:
            raise ValueError("unsupported Pareto scope")
        if not self.points:
            raise ValueError("Pareto section needs at least one point")
        if self.dominated_count != sum(1 for item in self.points if item.get("dominated")):
            raise ValueError("dominated count must match the marked points")

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "point_count": len(self.points),
            "dominated_count": self.dominated_count,
            "points": [dict(item) for item in self.points],
        }


def build_pareto_section(
    rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    scope: str | None = None,
) -> ParetoSection:
    """Mark quality/latency/RSS points; fixture rows stay non-promotable."""

    values = tuple(rows) if rows is not None else default_pareto_rows()
    marked = pareto_points(values)
    return ParetoSection(
        scope=scope or ("fixture" if rows is None else "live_loopback"),
        points=marked,
        dominated_count=sum(1 for item in marked if item.get("dominated")),
    )


@dataclass(frozen=True, slots=True)
class EvidenceEntry:
    """One ticket's closeout evidence with an explicit non-claim."""

    ticket: str
    theme: str
    status: str
    key_numbers: str
    evidence_paths: tuple[str, ...]
    scope: str
    boundary: str

    def __post_init__(self) -> None:
        if not self.ticket.strip() or not self.theme.strip():
            raise ValueError("ledger ticket and theme are required")
        if self.status not in {"completed", "in_progress", "planned"}:
            raise ValueError("unsupported ledger status")
        if self.scope not in EVIDENCE_SCOPES:
            raise ValueError("unsupported evidence scope")
        if not self.boundary.strip():
            raise ValueError("ledger entries must state their boundary")
        for path in self.evidence_paths:
            if not path.strip() or path.startswith(("/", "\\")) or _drive_letter.match(path) or ".." in path.replace("\\", "/").split("/"):
                raise ValueError("evidence paths must be repository-relative")

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticket": self.ticket,
            "theme": self.theme,
            "status": self.status,
            "key_numbers": self.key_numbers,
            "evidence_paths": list(self.evidence_paths),
            "scope": self.scope,
            "boundary": self.boundary,
        }


_drive_letter = re.compile(r"^[A-Za-z]:")


@dataclass(frozen=True, slots=True)
class S5CloseReport:
    """Closeout report: four sections, one digest, no unstated claims."""

    contract_drift: ContractDriftSection
    ollama: OllamaComparisonSection
    pareto: ParetoSection
    ledger: tuple[EvidenceEntry, ...]
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError("unsupported S5 closeout schema")
        if not self.ledger:
            raise ValueError("closeout report needs an evidence ledger")

    def digest_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "sections": list(SECTIONS),
            "contract_drift": self.contract_drift.as_dict(),
            "ollama": {
                "status": self.ollama.status,
                "ollama_model": self.ollama.ollama_model,
                "baseline_model": self.ollama.baseline_model,
                "agreement_rate": round(self.ollama.agreement_rate, 6),
                "entries": [
                    {
                        "prompt_id": item.prompt_id,
                        "agreement": item.agreement,
                        "ollama_tokens": item.ollama_tokens,
                        "baseline_tokens": item.baseline_tokens,
                    }
                    for item in self.ollama.entries
                ],
            },
            "pareto": {
                "scope": self.pareto.scope,
                "points": [dict(item) for item in self.pareto.points],
            },
            "ledger": [item.as_dict() for item in self.ledger],
        }

    @property
    def digest(self) -> str:
        return _digest(self.digest_payload())

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "sections": list(SECTIONS),
            "contract_drift": self.contract_drift.as_dict(),
            "ollama_comparison": self.ollama.as_dict(),
            "pareto": self.pareto.as_dict(),
            "evidence_ledger": [item.as_dict() for item in self.ledger],
            "network_used": False,
            "weights_loaded": self.ollama.status == "completed",
        }
        if include_digest:
            value["report_digest"] = self.digest
        return value

    def to_markdown(self) -> str:
        lines = [
            "# S5-CLOSE-01 Koakumix closeout report",
            "",
            f"- Sections: {', '.join(SECTIONS)}",
            f"- Contract drift: {self.contract_drift.matched} matched / {self.contract_drift.drifted} drifted / "
            f"{self.contract_drift.failed} failed (runner `{self.contract_drift.runner_kind}`)",
            f"- Ollama comparison: `{self.ollama.status}`"
            + (
                f" (agreement {self.ollama.agreement_rate:.3f}, latency ratio {self.ollama.latency_ratio:.3f})"
                if self.ollama.status == "completed"
                else f" ({self.ollama.reason})"
            ),
            f"- Pareto: {len(self.pareto.points)} points, {self.pareto.dominated_count} dominated (scope `{self.pareto.scope}`)",
            "- Latency fields are excluded from the report digest.",
            "",
            "## Contract drift",
            "",
            "| case | mismatch |",
            "| --- | --- |",
        ]
        for case_id, mismatch in self.contract_drift.drift_cases or (("(none)", "no drift detected"),):
            lines.append(f"| `{case_id}` | `{mismatch}` |")
        lines.extend(
            (
                "",
                "## Ollama comparison",
                "",
            )
        )
        if self.ollama.status == "completed":
            lines.extend(
                (
                    f"- Ollama `{self.ollama.ollama_model}` vs llama-server `{self.ollama.baseline_model}`",
                    "",
                    "| prompt | agreement | ollama tokens | baseline tokens | ollama ms | baseline ms |",
                    "| --- | --- | ---: | ---: | ---: | ---: |",
                )
            )
            for item in self.ollama.entries:
                lines.append(
                    f"| `{item.prompt_id}` | {'same' if item.agreement else 'different'} | "
                    f"{item.ollama_tokens} | {item.baseline_tokens} | {item.ollama_latency_ms} | {item.baseline_latency_ms} |"
                )
        else:
            lines.append(f"- Not run: `{self.ollama.reason}` (inject transports to execute).")
        lines.extend(
            (
                "",
                "## Pareto marking",
                "",
                "| point | quality | latency p95 (ms) | rss peak (bytes) | dominated |",
                "| --- | ---: | ---: | ---: | --- |",
            )
        )
        for point in self.pareto.points:
            lines.append(
                f"| `{point['id']}` | {float(point['quality_rate']):.3f} | "
                f"{float(point['latency_p95_ms']):.0f} | {int(point['rss_peak_bytes'])} | "
                f"{'yes' if point.get('dominated') else 'no'} |"
            )
        lines.extend(
            (
                "",
                "## Evidence ledger",
                "",
                "| ticket | status | scope | key numbers | evidence | boundary |",
                "| --- | --- | --- | --- | --- | --- |",
            )
        )
        for entry in self.ledger:
            lines.append(
                f"| `{entry.ticket}` | {entry.status} | `{entry.scope}` | {entry.key_numbers} | "
                f"{', '.join('`' + path + '`' for path in entry.evidence_paths)} | {entry.boundary} |"
            )
        lines.extend(("", f"Report digest: `{self.digest}`", ""))
        return "\n".join(lines)


def run_s5_close(
    *,
    api_report: APIWorkbenchReport | None = None,
    pareto_rows: Sequence[Mapping[str, Any]] | None = None,
    ollama_config: OllamaComparisonConfig | None = None,
    ollama_transport: ChatTransport | None = None,
    baseline_transport: ChatTransport | None = None,
    ledger: Sequence[EvidenceEntry] | None = None,
) -> S5CloseReport:
    """Build the closeout report; only injected transports touch a server."""

    return S5CloseReport(
        contract_drift=build_contract_drift_section(api_report),
        ollama=run_ollama_comparison(
            ollama_config,
            ollama_transport=ollama_transport,
            baseline_transport=baseline_transport,
        ),
        pareto=build_pareto_section(pareto_rows),
        ledger=tuple(ledger) if ledger is not None else default_evidence_ledger(),
    )


__all__ = [
    "DEFAULT_BASELINE_ENDPOINT",
    "DEFAULT_BASELINE_MODEL",
    "DEFAULT_CLOSE_PROMPTS",
    "DEFAULT_OLLAMA_ENDPOINT",
    "DEFAULT_OLLAMA_MODEL",
    "EVIDENCE_SCOPES",
    "S5_CLOSE_SCHEMA",
    "SECTIONS",
    "ContractDriftSection",
    "EvidenceEntry",
    "LocalBackendError",
    "OllamaComparisonConfig",
    "OllamaComparisonEntry",
    "OllamaComparisonSection",
    "ParetoSection",
    "S5CloseReport",
    "UrllibChatTransport",
    "build_contract_drift_section",
    "build_pareto_section",
    "default_evidence_ledger",
    "default_pareto_rows",
    "run_ollama_comparison",
    "run_s5_close",
]


S5_CLOSE_SCHEMA = SCHEMA
