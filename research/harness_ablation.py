"""Offline three-arm ablation contract for ``EX-HARNESS-01``.

The Koakumix value claim is "the same model answers better inside the harness
than in plain chat".  This module fixes the measurement contract for that
claim: one deterministic task set, three arms (``baseline`` plain chat,
``rag_memory`` retrieval plus long-term memory, ``compress_roles`` STATE
compression plus memory plus a draft/verify role plan), and four auditable
metrics (over-budget multi-turn success, cross-session memory recall, RAG
citation accuracy, and input-token cost).

Everything here is fixture-only: no model weights, no network, no user paths.
The report proves the pipeline and the metric definitions are connected and
reproducible; it is **not** evidence about model answer quality.  The real
three-round calibration runs against a live backend through the same fields.
"""

from __future__ import annotations

import gc
import hashlib
import json
import re
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Sequence

from ..context_engine import ContextBudget, ContextMessage, ContextPolicy, ContextPolicyConfig
from ..context_engine.tokenizer import HeuristicTokenizer, TokenCounter, count_message
from ..memory import MemoryStore
from ..rag import RagStore, build_context


SCHEMA = "qlh.harness.ablation.v1"
ARMS = ("baseline", "rag_memory", "compress_roles")
ARM_ENHANCEMENTS: dict[str, tuple[str, ...]] = {
    "baseline": ("plain_chat",),
    "rag_memory": ("retrieval", "long_term_memory"),
    "compress_roles": ("state_compression", "long_term_memory", "role_plan"),
}
TASK_KINDS = ("over_budget_multi_turn", "cross_session_memory", "rag_citation")
MEMORY_KINDS = ("fact", "preference", "decision")
DEFAULT_ROUNDS = 3
DEFAULT_INPUT_BUDGET = 192
OWNER_SCOPE = "ablation"
ROLE_PLAN_TEXT = (
    "Role plan: the draft role writes a first pass and the verify role checks it "
    "before the answer is returned."
)
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]+$")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class AblationTask:
    """One fixed task with its success predicate and optional fixtures."""

    task_id: str
    kind: str
    messages: tuple[ContextMessage, ...]
    required_facts: tuple[str, ...]
    query: str
    retrieval_query: str = ""
    prior_session_facts: tuple[str, ...] = ()
    memory_kind: str = "fact"
    rag_source_ref: str | None = None
    rag_source_text: str | None = None
    rag_target_phrase: str | None = None

    def __post_init__(self) -> None:
        if not self.task_id or not _IDENTIFIER.fullmatch(self.task_id):
            raise ValueError("task id is invalid")
        if self.kind not in TASK_KINDS:
            raise ValueError("unsupported ablation task kind")
        if not self.messages:
            raise ValueError("task needs at least one message")
        if not self.required_facts or any(not item.strip() for item in self.required_facts):
            raise ValueError("task needs at least one non-empty required fact")
        if not self.query.strip():
            raise ValueError("task query is required")
        if self.retrieval_query and not self.retrieval_query.strip():
            raise ValueError("retrieval query must be non-empty when declared")
        if self.memory_kind not in MEMORY_KINDS:
            raise ValueError("unsupported memory kind")
        cross_session = self.kind == "cross_session_memory"
        if cross_session and not self.prior_session_facts:
            raise ValueError("cross-session tasks need prior-session facts")
        if not cross_session and self.prior_session_facts:
            raise ValueError("only cross-session tasks may declare prior-session facts")
        rag = self.kind == "rag_citation"
        declared = (self.rag_source_ref, self.rag_source_text, self.rag_target_phrase)
        if rag and any(item is None for item in declared):
            raise ValueError("rag citation tasks need a source reference, text, and target phrase")
        if not rag and any(item is not None for item in declared):
            raise ValueError("only rag citation tasks may declare a rag source")

    @property
    def search_query(self) -> str:
        """Keyword-style retrieval query; the live pipeline derives it by rewriting."""

        return self.retrieval_query or self.query

    @property
    def digest(self) -> str:
        return _digest(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "messages": [message.as_dict() for message in self.messages],
            "required_facts": list(self.required_facts),
            "query": self.query,
            "retrieval_query": self.retrieval_query,
            "prior_session_facts": list(self.prior_session_facts),
            "memory_kind": self.memory_kind,
            "rag_source_ref": self.rag_source_ref,
            "rag_source_text": self.rag_source_text,
            "rag_target_phrase": self.rag_target_phrase,
        }


@dataclass(frozen=True, slots=True)
class HarnessAblationFixture:
    """A deterministic task set covering all three measurement families."""

    id: str
    tasks: tuple[AblationTask, ...]
    rounds: int = DEFAULT_ROUNDS
    seed: int = 17
    version: str = "v1"

    def __post_init__(self) -> None:
        if not self.id or not _IDENTIFIER.fullmatch(self.id):
            raise ValueError("fixture id is invalid")
        if self.rounds < DEFAULT_ROUNDS:
            raise ValueError("ablation fixture needs at least three rounds")
        if self.seed < 0 or not self.version:
            raise ValueError("fixture seed and version are required")
        if not self.tasks:
            raise ValueError("fixture needs at least one task")
        ids = [task.task_id for task in self.tasks]
        if len(set(ids)) != len(ids):
            raise ValueError("task ids must be unique")
        if {task.kind for task in self.tasks} != set(TASK_KINDS):
            raise ValueError("fixture must cover every ablation task kind")

    @property
    def digest(self) -> str:
        return _digest(self.as_dict(include_digest=False))

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "id": self.id,
            "tasks": [task.as_dict() for task in self.tasks],
            "rounds": self.rounds,
            "seed": self.seed,
            "version": self.version,
        }
        if include_digest:
            value["fixture_digest"] = self.digest
        return value


def _conversation(rounds: int, *, prefix: str, opener: str, question: str) -> tuple[ContextMessage, ...]:
    messages: list[ContextMessage] = []
    for round_id in range(rounds):
        if round_id == 0:
            content = opener
        elif round_id == rounds - 1:
            content = question
        else:
            content = (
                f"Round {round_id:02d} follow-up: review implementation detail {round_id} "
                f"and confirm nothing else changed."
            )
        messages.append(ContextMessage("user", content, f"{prefix}-u-{round_id:02d}", round_id))
        messages.append(
            ContextMessage(
                "assistant",
                f"Round {round_id:02d} acknowledged; detail {round_id} is unchanged.",
                f"{prefix}-a-{round_id:02d}",
                round_id,
            )
        )
    return tuple(messages)


def build_harness_ablation_fixture(*, seed: int = 17) -> HarnessAblationFixture:
    """Build the shared task set used by every arm."""

    early_fact = "Fact: the staging gate codename recorded today is INDIGO-LANTERN."
    constraint = "Fact: the hard constraint for this session is never to reveal internal build numbers."
    cross_fact = "Preference: the user always wants metric units in every answer."
    rag_target = "Omission counters must be reported instead of silent truncation"
    tasks = (
        AblationTask(
            task_id="over-budget-early-fact",
            kind="over_budget_multi_turn",
            messages=_conversation(
                30,
                prefix="ab-early",
                opener=early_fact,
                question="Which staging gate codename was recorded at the start of this session?",
            ),
            required_facts=(early_fact,),
            query="staging gate codename INDIGO-LANTERN",
        ),
        AblationTask(
            task_id="over-budget-constraint",
            kind="over_budget_multi_turn",
            messages=_conversation(
                24,
                prefix="ab-constraint",
                opener=constraint,
                question="Which hard constraint was set at the start of this session?",
            ),
            required_facts=(constraint,),
            query="hard constraint internal build numbers",
        ),
        AblationTask(
            task_id="cross-session-units",
            kind="cross_session_memory",
            messages=_conversation(
                8,
                prefix="ab-cross",
                opener="Round 00 follow-up: continue the review in a fresh session.",
                question="Which unit system did I ask for earlier?",
            ),
            required_facts=(cross_fact,),
            query="which unit system did the user ask for",
            retrieval_query="metric units",
            prior_session_facts=(cross_fact,),
            memory_kind="preference",
        ),
        AblationTask(
            task_id="rag-budget-omissions",
            kind="rag_citation",
            messages=_conversation(
                8,
                prefix="ab-rag",
                opener="Round 00 follow-up: start the budget review from the stored notes.",
                question="How are omissions reported in the layered budget?",
            ),
            required_facts=(rag_target,),
            query="how are omissions reported in the layered budget",
            retrieval_query="silent truncation layered budget",
            rag_source_ref="fixtures/ablation/layered-budget-notes.md",
            rag_source_text=(
                "Layered budget notes. The budget keeps memory, retrieval, and recent context "
                "inside one input budget. Omission counters must be reported instead of silent "
                "truncation, and truncated results carry an explicit flag."
            ),
            rag_target_phrase=rag_target,
        ),
    )
    return HarnessAblationFixture(id="ablation-core-v1", tasks=tasks, seed=seed)


@dataclass(frozen=True, slots=True)
class HarnessAblationObservation:
    """One arm/task/round cell with only auditable, non-model metrics."""

    arm: str
    task_id: str
    task_kind: str
    round_index: int
    input_budget: int
    input_tokens: int
    state_tokens: int
    memory_tokens: int
    rag_tokens: int
    recent_tokens: int
    retrieval_included: int
    memory_recall_hits: int
    memory_recall_expected: int
    citation_expected: bool
    citation_correct: bool
    task_success: bool
    compression_strategy: str
    fixture_digest: str
    seed: int
    runner_kind: str = "fixture"
    network_used: bool = False
    weights_loaded: bool = False
    notices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError("unsupported ablation arm")
        if self.task_kind not in TASK_KINDS:
            raise ValueError("unsupported ablation task kind")
        if self.round_index < 0 or self.input_budget <= 0 or self.input_tokens < 0:
            raise ValueError("round index, budget, and token counts are invalid")
        if not 0 <= self.memory_recall_hits <= self.memory_recall_expected:
            raise ValueError("memory recall counts are invalid")
        if not self.citation_expected and self.citation_correct:
            raise ValueError("citation correctness requires an expected citation")
        counts = (
            self.state_tokens,
            self.memory_tokens,
            self.rag_tokens,
            self.recent_tokens,
            self.retrieval_included,
            self.memory_recall_expected,
        )
        if any(value < 0 for value in counts):
            raise ValueError("measurement counts must be non-negative")
        if not re.fullmatch(r"[0-9a-f]{64}", self.fixture_digest) or self.seed < 0:
            raise ValueError("fixture digest and seed are invalid")
        if self.runner_kind != "fixture" or self.network_used or self.weights_loaded:
            raise ValueError("ablation observations are fixture-only and model-free")

    @property
    def memory_recall_rate(self) -> float:
        if self.memory_recall_expected == 0:
            return 0.0
        return self.memory_recall_hits / self.memory_recall_expected

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "task_id": self.task_id,
            "task_kind": self.task_kind,
            "round_index": self.round_index,
            "input_budget": self.input_budget,
            "input_tokens": self.input_tokens,
            "state_tokens": self.state_tokens,
            "memory_tokens": self.memory_tokens,
            "rag_tokens": self.rag_tokens,
            "recent_tokens": self.recent_tokens,
            "retrieval_included": self.retrieval_included,
            "memory_recall_hits": self.memory_recall_hits,
            "memory_recall_expected": self.memory_recall_expected,
            "memory_recall_rate": self.memory_recall_rate,
            "citation_expected": self.citation_expected,
            "citation_correct": self.citation_correct,
            "task_success": self.task_success,
            "compression_strategy": self.compression_strategy,
            "fixture_digest": self.fixture_digest,
            "seed": self.seed,
            "runner_kind": self.runner_kind,
            "network_used": self.network_used,
            "weights_loaded": self.weights_loaded,
            "notices": list(self.notices),
        }


@dataclass(frozen=True, slots=True)
class HarnessAblationArmSummary:
    """Aggregated per-arm metrics, cost ratio, and round stability."""

    arm: str
    enhancements: tuple[str, ...]
    task_count: int
    rounds: int
    observation_count: int
    task_success_rate: float
    memory_recall_rate: float
    citation_accuracy: float
    citation_attempts: int
    input_tokens_mean: float
    input_tokens_total: int
    token_cost_ratio: float
    stability_margin: float
    stable: bool

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError("unsupported ablation arm")
        if self.task_count <= 0 or self.rounds < DEFAULT_ROUNDS:
            raise ValueError("summary needs at least one task and three rounds")
        if self.observation_count != self.task_count * self.rounds:
            raise ValueError("observation count must match tasks times rounds")
        for value in (self.task_success_rate, self.memory_recall_rate, self.citation_accuracy):
            if not 0.0 <= value <= 1.0:
                raise ValueError("rates must be between zero and one")
        if self.token_cost_ratio < 0 or self.stability_margin < 0:
            raise ValueError("cost ratio and stability margin must be non-negative")
        if self.input_tokens_mean < 0 or self.input_tokens_total < 0:
            raise ValueError("token counts must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "enhancements": list(self.enhancements),
            "task_count": self.task_count,
            "rounds": self.rounds,
            "observation_count": self.observation_count,
            "task_success_rate": self.task_success_rate,
            "memory_recall_rate": self.memory_recall_rate,
            "citation_accuracy": self.citation_accuracy,
            "citation_attempts": self.citation_attempts,
            "input_tokens_mean": self.input_tokens_mean,
            "input_tokens_total": self.input_tokens_total,
            "token_cost_ratio": self.token_cost_ratio,
            "stability_margin": self.stability_margin,
            "stable": self.stable,
        }


@dataclass(frozen=True, slots=True)
class HarnessAblationReport:
    """Three-arm comparison envelope with reproducible digests."""

    fixture: HarnessAblationFixture
    observations: tuple[HarnessAblationObservation, ...]
    summaries: tuple[HarnessAblationArmSummary, ...]
    input_budget: int = DEFAULT_INPUT_BUDGET
    seed: int = 17
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError("unsupported ablation schema")
        if self.input_budget <= 0:
            raise ValueError("input budget must be positive")
        expected = len(ARMS) * len(self.fixture.tasks) * self.fixture.rounds
        if len(self.observations) != expected:
            raise ValueError("one observation is required for every arm/task/round cell")
        cells = {(item.arm, item.task_id, item.round_index) for item in self.observations}
        if len(cells) != expected:
            raise ValueError("observations must contain one unique cell per arm, task, and round")
        if {item.arm for item in self.summaries} != set(ARMS):
            raise ValueError("every arm requires a summary")
        if self.fixture.seed != self.seed:
            raise ValueError("fixture seed must match report seed")
        if any(
            item.fixture_digest != self.fixture.digest
            or item.seed != self.seed
            or item.input_budget != self.input_budget
            for item in self.observations
        ):
            raise ValueError("observation provenance does not match report")

    @property
    def digest(self) -> str:
        return _digest(self.as_dict(include_digest=False))

    def arm_rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(summary.as_dict() for summary in self.summaries)

    def series(self) -> dict[str, tuple[dict[str, Any], ...]]:
        return {
            arm: tuple(
                {
                    "round_index": item.round_index,
                    "task_id": item.task_id,
                    "task_success": item.task_success,
                    "memory_recall_rate": item.memory_recall_rate,
                    "citation_correct": item.citation_correct,
                    "input_tokens": item.input_tokens,
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
            "input_budget": self.input_budget,
            "observations": [item.as_dict() for item in self.observations],
            "summaries": [summary.as_dict() for summary in self.summaries],
            "series": {key: list(items) for key, items in self.series().items()},
            "seed": self.seed,
            "runner_kind": "fixture",
            "network_used": False,
            "weights_loaded": False,
            "evidence_scope": "pipeline_and_metric_contract_only",
        }
        if include_digest:
            value["report_digest"] = self.digest
        return value

    def to_markdown(self) -> str:
        """Render the three-arm comparison without claiming model quality."""

        lines = [
            "# EX-HARNESS-01 Koakumix enhancement ablation (fixture runner)",
            "",
            f"- Fixture: `{self.fixture.id}` ({len(self.fixture.tasks)} tasks, "
            f"{self.fixture.rounds} rounds per arm, `{self.fixture.digest[:12]}`)",
            f"- Input budget: {self.input_budget}; seed: {self.seed}",
            "- Runner: `fixture`; weights loaded: `false`; network used: `false`",
            "- This table proves the pipeline and metric definitions, not model answer quality.",
            "",
            "| arm | enhancements | task success | memory recall | citation accuracy | input tokens (mean) | cost ratio | stable |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for summary in self.summaries:
            lines.append(
                f"| `{summary.arm}` | {', '.join(summary.enhancements)} | "
                f"{summary.task_success_rate:.3f} | {summary.memory_recall_rate:.3f} | "
                f"{summary.citation_accuracy:.3f} | {summary.input_tokens_mean:.1f} | "
                f"{summary.token_cost_ratio:.3f} | {'yes' if summary.stable else 'no'} |"
            )
        lines.extend(
            (
                "",
                "## Per-task outcome",
                "",
                "| arm | task | kind | success | recall | citation |",
                "| --- | --- | --- | --- | ---: | ---: |",
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
                f"{'pass' if item.task_success else 'miss'} | {item.memory_recall_hits}/{item.memory_recall_expected} | "
                f"{'correct' if item.citation_correct else ('not-expected' if not item.citation_expected else 'wrong')} |"
            )
        lines.extend(
            (
                "",
                "## Real-run handoff",
                "",
                "The three-round real calibration reuses these arms, tasks, and metric fields; "
                "`runner_kind` becomes `llama_server` (or `qlh`) only after weights are loaded and "
                "the run is recorded with the same report schema.",
                "",
                f"Report digest: `{self.digest}`",
            )
        )
        return "\n".join(lines) + "\n"


def _budget(input_budget: int) -> ContextBudget:
    return ContextBudget(n_ctx=input_budget + 128, max_new_tokens=64, overhead=64)


def _arm_policy(arm: str, tokenizer: TokenCounter) -> ContextPolicy:
    if arm == "baseline":
        config = ContextPolicyConfig(
            recent_turns=4,
            summary_trigger_ratio=1.0,
            recent_turn_ratio=0.60,
            compression_strategy="mask",
            memory_recall_ratio=0.0,
            memory_recall_limit=0,
        )
    elif arm == "rag_memory":
        config = ContextPolicyConfig(
            recent_turns=4,
            summary_trigger_ratio=1.0,
            recent_turn_ratio=0.60,
            compression_strategy="mask",
            memory_recall_ratio=0.30,
            memory_recall_limit=8,
        )
    elif arm == "compress_roles":
        config = ContextPolicyConfig(
            recent_turns=4,
            summary_trigger_ratio=0.70,
            recent_turn_ratio=0.35,
            compression_strategy="state",
            state_variant="compact",
            memory_recall_ratio=0.30,
            memory_recall_limit=8,
        )
    else:
        raise ValueError("unsupported ablation arm")
    return ContextPolicy(config=config, tokenizer=tokenizer)


def _count_kind(messages: Iterable[ContextMessage], kind: str, tokenizer: TokenCounter) -> int:
    return sum(count_message(tokenizer, message) for message in messages if message.kind == kind)


@dataclass(frozen=True, slots=True)
class ArmAssembly:
    """One arm's assembled model input plus its audit metrics.

    ``assembled_text`` is what a live backend would receive; it never enters
    ``metrics()``, which keeps only counts and a SHA-256 of the input.
    """

    arm: str
    task_id: str
    assembled_text: str
    input_tokens: int
    state_tokens: int
    memory_tokens: int
    rag_tokens: int
    recent_tokens: int
    retrieval_included: int
    memory_recall_hits: int
    memory_recall_expected: int
    citation_expected: bool
    citation_correct: bool
    fixture_task_success: bool
    compression_strategy: str
    notices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError("unsupported ablation arm")
        if not self.assembled_text.strip():
            raise ValueError("assembled text is required")
        counts = (
            self.input_tokens,
            self.state_tokens,
            self.memory_tokens,
            self.rag_tokens,
            self.recent_tokens,
            self.retrieval_included,
            self.memory_recall_expected,
        )
        if any(value < 0 for value in counts):
            raise ValueError("assembly counts must be non-negative")
        if not 0 <= self.memory_recall_hits <= self.memory_recall_expected:
            raise ValueError("memory recall counts are invalid")
        if not self.citation_expected and self.citation_correct:
            raise ValueError("citation correctness requires an expected citation")

    @property
    def memory_recall_rate(self) -> float:
        if self.memory_recall_expected == 0:
            return 0.0
        return self.memory_recall_hits / self.memory_recall_expected

    def metrics(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "task_id": self.task_id,
            "input_tokens": self.input_tokens,
            "state_tokens": self.state_tokens,
            "memory_tokens": self.memory_tokens,
            "rag_tokens": self.rag_tokens,
            "recent_tokens": self.recent_tokens,
            "retrieval_included": self.retrieval_included,
            "memory_recall_hits": self.memory_recall_hits,
            "memory_recall_expected": self.memory_recall_expected,
            "memory_recall_rate": self.memory_recall_rate,
            "citation_expected": self.citation_expected,
            "citation_correct": self.citation_correct,
            "fixture_task_success": self.fixture_task_success,
            "compression_strategy": self.compression_strategy,
            "assembled_chars": len(self.assembled_text),
            "assembled_sha256": hashlib.sha256(self.assembled_text.encode("utf-8")).hexdigest(),
            "notices": list(self.notices),
        }


def assemble_arm_context(
    task: AblationTask,
    arm: str,
    input_budget: int,
    *,
    tokenizer: TokenCounter,
) -> ArmAssembly:
    """Assemble one arm's model input without loading weights or touching the network."""

    policy = _arm_policy(arm, tokenizer)
    use_memory = arm != "baseline"
    use_rag = arm == "rag_memory"
    temporary = TemporaryDirectory(prefix="qlh-ablation-")
    directory = temporary.name
    store: MemoryStore | None = None
    rag_store: RagStore | None = None
    snapshot = None
    retrieval = None
    try:
        store = MemoryStore(f"{directory}/memory.sqlite3") if use_memory else None
        if store is not None:
            for index, fact in enumerate(task.prior_session_facts):
                store.add(
                    kind=task.memory_kind,  # type: ignore[arg-type]
                    content=fact,
                    owner_scope=OWNER_SCOPE,
                    source_session_id=f"session-a-{task.task_id}",
                    source_message_id=f"session-a-{task.task_id}-{index}",
                )
        if use_rag and task.rag_source_ref and task.rag_source_text:
            rag_store = RagStore(f"{directory}/rag.sqlite3")
            rag_store.add_document(
                source_ref=task.rag_source_ref,
                title="ablation-fixture-source",
                text=task.rag_source_text,
                owner_scope=OWNER_SCOPE,
            )
            hits = rag_store.search(task.search_query, owner_scope=OWNER_SCOPE, limit=3)
            retrieval = build_context(
                [hit.as_dict() for hit in hits],
                max_chars=4_000,
                tokenizer=tokenizer.count,
            )
        snapshot = policy.build(
            task.messages,
            _budget(input_budget),
            memory_store=store,
            memory_owner_scope=OWNER_SCOPE,
            memory_source_session_id=f"session-b-{task.task_id}",
            memory_query=(task.search_query if store is not None else None),
        )
        parts = [message.content for message in snapshot.messages]
        if arm == "compress_roles":
            parts.append(ROLE_PLAN_TEXT)
        if retrieval is not None and retrieval.text:
            parts.append(retrieval.text)
        assembled = "\n".join(parts)
        recall_hits = sum(1 for fact in task.prior_session_facts if fact in assembled)
        citation_expected = task.rag_target_phrase is not None
        citation_correct = bool(
            citation_expected
            and retrieval is not None
            and retrieval.included_count > 0
            and task.rag_target_phrase in retrieval.text
        )
        assembly = ArmAssembly(
            arm=arm,
            task_id=task.task_id,
            assembled_text=assembled,
            input_tokens=tokenizer.count(assembled),
            state_tokens=_count_kind(snapshot.messages, "summary", tokenizer),
            memory_tokens=_count_kind(snapshot.messages, "memory", tokenizer),
            rag_tokens=int(retrieval.token_count) if retrieval is not None else 0,
            recent_tokens=sum(
                count_message(tokenizer, message)
                for message in snapshot.messages
                if message.kind not in {"summary", "memory", "verbatim"}
            ),
            retrieval_included=int(retrieval.included_count) if retrieval is not None else 0,
            memory_recall_hits=recall_hits,
            memory_recall_expected=len(task.prior_session_facts),
            citation_expected=citation_expected,
            citation_correct=citation_correct,
            fixture_task_success=all(fact in assembled for fact in task.required_facts),
            compression_strategy=snapshot.compression_strategy,
            notices=tuple(notice.code for notice in snapshot.notices),
        )
    finally:
        # sqlite3 connection contexts commit but do not close the handle; drop
        # the stores before TemporaryDirectory unlinks the files on Windows.
        del snapshot
        del retrieval
        del store
        del rag_store
        del policy
        gc.collect()
        temporary.cleanup()
    return assembly


def _run_cell(
    fixture: HarnessAblationFixture,
    task: AblationTask,
    arm: str,
    round_index: int,
    input_budget: int,
    *,
    tokenizer: TokenCounter,
) -> HarnessAblationObservation:
    assembly = assemble_arm_context(task, arm, input_budget, tokenizer=tokenizer)
    return HarnessAblationObservation(
        arm=assembly.arm,
        task_id=assembly.task_id,
        task_kind=task.kind,
        round_index=round_index,
        input_budget=input_budget,
        input_tokens=assembly.input_tokens,
        state_tokens=assembly.state_tokens,
        memory_tokens=assembly.memory_tokens,
        rag_tokens=assembly.rag_tokens,
        recent_tokens=assembly.recent_tokens,
        retrieval_included=assembly.retrieval_included,
        memory_recall_hits=assembly.memory_recall_hits,
        memory_recall_expected=assembly.memory_recall_expected,
        citation_expected=assembly.citation_expected,
        citation_correct=assembly.citation_correct,
        task_success=assembly.fixture_task_success,
        compression_strategy=assembly.compression_strategy,
        fixture_digest=fixture.digest,
        seed=fixture.seed,
        notices=assembly.notices,
    )


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _summarize(
    arm: str,
    observations: tuple[HarnessAblationObservation, ...],
    *,
    baseline_tokens_mean: float,
) -> HarnessAblationArmSummary:
    cells = tuple(item for item in observations if item.arm == arm)
    rounds = sorted({item.round_index for item in cells})
    task_ids = sorted({item.task_id for item in cells})
    recall_cells = tuple(item for item in cells if item.memory_recall_expected > 0)
    per_round: list[tuple[float, float, float, float]] = []
    for round_index in rounds:
        round_cells = [item for item in cells if item.round_index == round_index]
        attempts = [item for item in round_cells if item.citation_expected]
        round_recall = [item for item in round_cells if item.memory_recall_expected > 0]
        per_round.append(
            (
                _mean([1.0 if item.task_success else 0.0 for item in round_cells]),
                _mean([item.memory_recall_rate for item in round_recall]),
                _mean([1.0 if item.citation_correct else 0.0 for item in attempts]),
                _mean([float(item.input_tokens) for item in round_cells]),
            )
        )
    margin = 0.0
    for column in range(4):
        values = [row[column] for row in per_round]
        margin = max(margin, max(values) - min(values))
    attempts = [item for item in cells if item.citation_expected]
    tokens_mean = _mean([float(item.input_tokens) for item in cells])
    return HarnessAblationArmSummary(
        arm=arm,
        enhancements=ARM_ENHANCEMENTS[arm],
        task_count=len(task_ids),
        rounds=len(rounds),
        observation_count=len(cells),
        task_success_rate=_mean([1.0 if item.task_success else 0.0 for item in cells]),
        memory_recall_rate=_mean([item.memory_recall_rate for item in recall_cells]),
        citation_accuracy=_mean([1.0 if item.citation_correct else 0.0 for item in attempts]),
        citation_attempts=len(attempts),
        input_tokens_mean=round(tokens_mean, 3),
        input_tokens_total=sum(item.input_tokens for item in cells),
        token_cost_ratio=round(tokens_mean / baseline_tokens_mean, 6) if baseline_tokens_mean else 0.0,
        stability_margin=round(margin, 9),
        stable=margin == 0.0,
    )


def run_harness_ablation(
    fixture: HarnessAblationFixture | None = None,
    *,
    input_budget: int = DEFAULT_INPUT_BUDGET,
    seed: int = 17,
    tokenizer: TokenCounter | None = None,
) -> HarnessAblationReport:
    """Run every arm/task/round cell with no model or network access."""

    fixture = fixture or build_harness_ablation_fixture(seed=seed)
    if fixture.seed != seed:
        raise ValueError("fixture seed must match ablation seed")
    if input_budget <= 0:
        raise ValueError("input budget must be positive")
    counter = tokenizer or HeuristicTokenizer()
    observations = tuple(
        _run_cell(fixture, task, arm, round_index, input_budget, tokenizer=counter)
        for arm in ARMS
        for task in fixture.tasks
        for round_index in range(fixture.rounds)
    )
    baseline_mean = _mean([float(item.input_tokens) for item in observations if item.arm == "baseline"])
    summaries = tuple(
        _summarize(arm, observations, baseline_tokens_mean=baseline_mean) for arm in ARMS
    )
    return HarnessAblationReport(
        fixture=fixture,
        observations=observations,
        summaries=summaries,
        input_budget=input_budget,
        seed=seed,
    )


def build_harness_ablation_report(
    fixture: HarnessAblationFixture | None = None,
    *,
    input_budget: int = DEFAULT_INPUT_BUDGET,
    seed: int = 17,
    tokenizer: TokenCounter | None = None,
) -> HarnessAblationReport:
    """Named builder alias for callers that prefer report-oriented APIs."""

    return run_harness_ablation(fixture, input_budget=input_budget, seed=seed, tokenizer=tokenizer)


__all__ = [
    "ABLATION_SCHEMA",
    "ARMS",
    "ARM_ENHANCEMENTS",
    "DEFAULT_INPUT_BUDGET",
    "DEFAULT_ROUNDS",
    "TASK_KINDS",
    "AblationTask",
    "ArmAssembly",
    "HarnessAblationArmSummary",
    "HarnessAblationFixture",
    "HarnessAblationObservation",
    "HarnessAblationReport",
    "assemble_arm_context",
    "build_harness_ablation_fixture",
    "build_harness_ablation_report",
    "run_harness_ablation",
]


ABLATION_SCHEMA = SCHEMA
