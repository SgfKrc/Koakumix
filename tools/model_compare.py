"""Offline multi-model output matrix for ``TOOL-MODEL-CMP-01``.

Renders one fixed prompt set through several registered models and records a
per-model answer matrix judged with the shared v2 ``loose_contains`` policy
(``TOOL-JUDGE-POLICY-01``).  The default provider is a fixture that uses the
rubric's accepted answers as placeholders and is labelled
``answer_source=rubric_placeholder``; real runs inject a provider and report
``runner_kind=injected``.

The rubric only covers the objectively judgeable subset of the prompt set, so
prompts without a rubric entry stay in the matrix as ``unrated`` cells and the
per-model ``correct_rate`` is computed over rated prompts only.

Answer text never enters the report: only SHA-256, character count, verdict,
and (for injected runs) latency.  Fixture runs keep ``latency_ms=0`` instead of
inventing timings, and a fixture matrix is an upper bound of the judging
contract — it is not a model-quality claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .judge_policy import V2_POLICY, JudgePolicy, JudgeRubric, evaluate_completion, load_judge_rubric


MODEL_COMPARE_SCHEMA = "qlh.harness.model_compare.v1"
MODEL_COMPARE_INPUT_SCHEMA = "qlh.model_compare.v1"
DEFAULT_MODEL_IDS: tuple[str, ...] = (
    "DistilQwen2.5-DS3-0324-7B",
    "Qwen3-4B",
    "DeepSeek-R1-Distill-Qwen-7B",
    "QW1.8B",
)
DEFAULT_PROMPT_SET = "fixtures/prompt_sets/ps-qwen3-nothink/prompts.jsonl"
DEFAULT_RUBRIC = "fixtures/quality_rubrics/llm-objective-ps-v1-qwen3-nothink-v1.json"
PLACEHOLDER_ANSWER_SOURCE = "rubric_placeholder"
INJECTED_ANSWER_SOURCE = "injected"
UNRATED_ANSWER_SOURCE = "unrated"
VERDICTS = ("passed", "failed", "missing", "truncated", "unrated")
MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]+$")
_ABSOLUTE_PATH = re.compile(r"(?:^[A-Za-z]:[\\/]|^/|^\\\\)")

AnswerProvider = Callable[[str, "ComparePrompt"], "str | None"]


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _relative_ref(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or _ABSOLUTE_PATH.search(value) or ".." in value.replace("\\", "/").split("/"):
        raise ValueError("prompt set reference must be repository-relative")
    return value.replace("\\", "/")


@dataclass(frozen=True, slots=True)
class ComparePrompt:
    """One fixed prompt taken from the shared prompt set."""

    prompt_id: str
    prompt_text: str

    def __post_init__(self) -> None:
        if not self.prompt_id.strip() or not self.prompt_text.strip():
            raise ValueError("compare prompts need an id and text")

    def as_dict(self) -> dict[str, Any]:
        return {"prompt_id": self.prompt_id, "prompt_chars": len(self.prompt_text)}


@dataclass(frozen=True, slots=True)
class ComparePromptSet:
    """A loaded prompt set with a stable content digest."""

    prompt_set_id: str
    source_ref: str
    sha256: str
    prompts: tuple[ComparePrompt, ...]

    def __post_init__(self) -> None:
        if not self.prompt_set_id.strip():
            raise ValueError("prompt set id is required")
        _relative_ref(self.source_ref)
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("prompt set digest must be a lowercase sha256")
        if not self.prompts or len({item.prompt_id for item in self.prompts}) != len(self.prompts):
            raise ValueError("prompt set entries must be non-empty and unique")

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_set_id": self.prompt_set_id,
            "source_ref": self.source_ref,
            "sha256": self.sha256,
            "prompt_count": len(self.prompts),
        }


def load_prompt_set(source: Path | str, *, prompt_set_id: str | None = None) -> ComparePromptSet:
    """Load a JSONL prompt set (``{"id": ..., "prompt": ...}`` per line)."""

    path = Path(source)
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise ValueError("prompt set is unavailable") from exc
    prompts: list[ComparePrompt] = []
    for line in raw_bytes.decode("utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError("prompt set contains invalid JSON lines") from exc
        if not isinstance(item, Mapping):
            raise ValueError("prompt set entries must be objects")
        prompt_id = item.get("id")
        prompt_text = item.get("prompt")
        if not isinstance(prompt_id, str) or not isinstance(prompt_text, str):
            raise ValueError("prompt set entries require string id and prompt")
        prompts.append(ComparePrompt(prompt_id, prompt_text))
    source_ref = path.as_posix()
    if _ABSOLUTE_PATH.search(source_ref):
        source_ref = path.name
    return ComparePromptSet(
        prompt_set_id=prompt_set_id or path.parent.name,
        source_ref=source_ref,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        prompts=tuple(prompts),
    )


def load_compare_rubric(source: Path | str, *, expected_sha256: str | None = None) -> JudgeRubric:
    """Load the shared ``correctness`` rubric with an optional SHA pin.

    ``judge_policy`` rubric entries are correctness-shaped; other dimensions of
    the same file (for example ``format``) use a different structure and are
    outside this tool's scope.
    """

    return load_judge_rubric(source, dimension="correctness", expected_sha256=expected_sha256)


def fixture_provider(rubric: JudgeRubric) -> AnswerProvider:
    """Placeholder provider: return the rubric's first accepted answer per prompt.

    This is deliberately an upper bound of the judging contract: every rated
    prompt maps to an accepted answer, so a fixture matrix passes on every
    rated cell.  Inject a real provider to obtain discriminating results.
    """

    accepted_by_prompt = {entry.prompt_id: entry.accepted[0] for entry in rubric.entries}

    def provider(model_id: str, prompt: ComparePrompt) -> str | None:
        del model_id
        return accepted_by_prompt.get(prompt.prompt_id)

    return provider


@dataclass(frozen=True, slots=True)
class ModelCell:
    """One prompt/model cell: verdict plus a hashed answer, never the text."""

    model_id: str
    prompt_id: str
    verdict: str
    answer_sha256: str
    answer_chars: int
    latency_ms: int
    answer_source: str

    def __post_init__(self) -> None:
        if not MODEL_ID_PATTERN.fullmatch(self.model_id) or not self.prompt_id.strip():
            raise ValueError("cell identity is invalid")
        if self.verdict not in VERDICTS:
            raise ValueError("unsupported verdict")
        if not re.fullmatch(r"[0-9a-f]{64}", self.answer_sha256):
            raise ValueError("answer digest must be a lowercase sha256")
        if self.answer_chars < 0 or self.latency_ms < 0:
            raise ValueError("cell counts must be non-negative")

    @property
    def passed(self) -> bool:
        return self.verdict == "passed"

    @property
    def rated(self) -> bool:
        return self.verdict in {"passed", "failed"}

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "prompt_id": self.prompt_id,
            "verdict": self.verdict,
            "answer_sha256": self.answer_sha256,
            "answer_chars": self.answer_chars,
            "latency_ms": self.latency_ms,
            "answer_source": self.answer_source,
        }


@dataclass(frozen=True, slots=True)
class ModelRow:
    """Aggregated row for one model across every prompt of the set."""

    model_id: str
    prompt_count: int
    rated_count: int
    unrated_count: int
    passed_count: int
    correct_rate: float
    latency_ms_mean: float
    verdict_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        if self.prompt_count <= 0:
            raise ValueError("row needs at least one prompt")
        if self.rated_count + self.unrated_count != self.prompt_count:
            raise ValueError("rated and unrated counts must cover every prompt")
        if not 0 <= self.passed_count <= self.rated_count:
            raise ValueError("passed count must fit the rated subset")
        if not 0.0 <= self.correct_rate <= 1.0:
            raise ValueError("correct rate must be between zero and one")
        if self.latency_ms_mean < 0:
            raise ValueError("latency must be non-negative")
        if sum(self.verdict_counts.values()) != self.prompt_count:
            raise ValueError("verdict counts must cover every prompt")

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "prompt_count": self.prompt_count,
            "rated_count": self.rated_count,
            "unrated_count": self.unrated_count,
            "passed_count": self.passed_count,
            "correct_rate": round(self.correct_rate, 6),
            "latency_ms_mean": round(self.latency_ms_mean, 1),
            "verdict_counts": dict(sorted(self.verdict_counts.items())),
        }


@dataclass(frozen=True, slots=True)
class ModelCompareReport:
    """Prompt-by-model answer matrix with the shared v2 judging policy."""

    prompt_set: ComparePromptSet
    model_ids: tuple[str, ...]
    rubric_id: str
    rubric_digest: str
    policy_name: str
    cells: tuple[ModelCell, ...]
    rows: tuple[ModelRow, ...]
    runner_kind: str = "fixture"
    answer_source: str = PLACEHOLDER_ANSWER_SOURCE
    schema: str = MODEL_COMPARE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != MODEL_COMPARE_SCHEMA:
            raise ValueError("unsupported model compare schema")
        if not self.model_ids or len(set(self.model_ids)) != len(self.model_ids):
            raise ValueError("model ids must be non-empty and unique")
        if self.runner_kind not in {"fixture", "injected"}:
            raise ValueError("unsupported runner kind")
        expected = len(self.model_ids) * len(self.prompt_set.prompts)
        if len(self.cells) != expected:
            raise ValueError("one cell is required for every model/prompt pair")
        pairs = {(cell.model_id, cell.prompt_id) for cell in self.cells}
        if len(pairs) != expected:
            raise ValueError("cells must be unique per model and prompt")
        if {cell.model_id for cell in self.cells} != set(self.model_ids):
            raise ValueError("every model requires cells")
        if {row.model_id for row in self.rows} != set(self.model_ids):
            raise ValueError("every model requires a row")
        if not re.fullmatch(r"[0-9a-f]{64}", self.rubric_digest):
            raise ValueError("rubric digest must be a lowercase sha256")

    def matrix(self) -> dict[str, dict[str, str]]:
        return {
            prompt.prompt_id: {
                cell.model_id: cell.verdict for cell in self.cells if cell.prompt_id == prompt.prompt_id
            }
            for prompt in self.prompt_set.prompts
        }

    @property
    def unrated_prompt_ids(self) -> tuple[str, ...]:
        matrix = self.matrix()
        return tuple(
            prompt.prompt_id
            for prompt in self.prompt_set.prompts
            if all(matrix[prompt.prompt_id].get(model_id) == "unrated" for model_id in self.model_ids)
        )

    def digest_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "prompt_set": self.prompt_set.as_dict(),
            "model_ids": list(self.model_ids),
            "rubric_id": self.rubric_id,
            "rubric_digest": self.rubric_digest,
            "policy_name": self.policy_name,
            "runner_kind": self.runner_kind,
            "answer_source": self.answer_source,
            "cells": [
                {
                    "model_id": cell.model_id,
                    "prompt_id": cell.prompt_id,
                    "verdict": cell.verdict,
                    "answer_sha256": cell.answer_sha256,
                    "answer_chars": cell.answer_chars,
                }
                for cell in self.cells
            ],
            "rows": [row.as_dict() for row in self.rows],
        }

    @property
    def digest(self) -> str:
        return _digest(self.digest_payload())

    @property
    def matrix_digest(self) -> str:
        return _digest(self.matrix())

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "prompt_set": self.prompt_set.as_dict(),
            "model_ids": list(self.model_ids),
            "rubric_id": self.rubric_id,
            "rubric_digest": self.rubric_digest,
            "policy_name": self.policy_name,
            "runner_kind": self.runner_kind,
            "answer_source": self.answer_source,
            "unrated_prompt_ids": list(self.unrated_prompt_ids),
            "cells": [cell.as_dict() for cell in self.cells],
            "rows": [row.as_dict() for row in self.rows],
            "matrix": {key: dict(items) for key, items in self.matrix().items()},
            "evidence_scope": (
                "pipeline_and_judging_contract_only"
                if self.runner_kind == "fixture"
                else "injected_provider_run"
            ),
            "weights_loaded": self.runner_kind == "injected",
            "network_used": False,
        }
        if include_digest:
            value["report_digest"] = self.digest
        return value

    def to_markdown(self) -> str:
        rated_prompts = len(self.prompt_set.prompts) - len(self.unrated_prompt_ids)
        lines = [
            "# TOOL-MODEL-CMP-01 multi-model output matrix",
            "",
            f"- Prompt set: `{self.prompt_set.prompt_set_id}` "
            f"({len(self.prompt_set.prompts)} prompts, `{self.prompt_set.sha256[:12]}`)",
            f"- Rubric: `{self.rubric_id}` (`{self.rubric_digest[:12]}`); policy: `{self.policy_name}`",
            f"- Rated prompts: {rated_prompts} of {len(self.prompt_set.prompts)} "
            "(rubric only covers the objectively judgeable subset; the rest stay `unrated`)",
            f"- Runner: `{self.runner_kind}`; answer source: `{self.answer_source}`; network used: `false`",
            "- Fixture runs use rubric placeholders: the matrix proves the pipeline and judging "
            "contract, not model quality.",
            "",
            "| prompt | " + " | ".join(f"`{model_id}`" for model_id in self.model_ids) + " |",
            "| --- | " + " | ".join("---" for _ in self.model_ids) + " |",
        ]
        matrix = self.matrix()
        for prompt in self.prompt_set.prompts:
            row = matrix[prompt.prompt_id]
            lines.append(
                "| `" + prompt.prompt_id + "` | "
                + " | ".join(row.get(model_id, "missing") for model_id in self.model_ids)
                + " |"
            )
        lines.extend(
            (
                "",
                "## Per-model summary",
                "",
                "| model | prompts | rated | passed | correct rate (rated) | verdicts | latency ms (mean) |",
                "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
            )
        )
        for row in self.rows:
            verdicts = ", ".join(f"{key}:{value}" for key, value in sorted(row.verdict_counts.items()))
            lines.append(
                f"| `{row.model_id}` | {row.prompt_count} | {row.rated_count} | {row.passed_count} | "
                f"{row.correct_rate:.3f} | {verdicts} | {row.latency_ms_mean:.1f} |"
            )
        lines.extend(
            (
                "",
                "## Real-run handoff",
                "",
                "Inject an answer provider (per-model transport) and the same schema records the real "
                "matrix with `runner_kind=injected`; judging stays on the shared v2 `loose_contains` policy.",
                "",
                f"Report digest: `{self.digest}`",
                "",
            )
        )
        return "\n".join(lines)


def run_model_compare(
    prompt_set: ComparePromptSet,
    rubric: JudgeRubric,
    *,
    model_ids: Sequence[str] = DEFAULT_MODEL_IDS,
    provider: AnswerProvider | None = None,
    policy: JudgePolicy = V2_POLICY,
    runner_kind: str | None = None,
) -> ModelCompareReport:
    """Build the prompt/model matrix; without a provider this is fixture-only."""

    ids = tuple(model_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("model ids must be non-empty and unique")
    for model_id in ids:
        if not MODEL_ID_PATTERN.fullmatch(model_id):
            raise ValueError("model id is invalid")
    rubric_by_prompt = {entry.prompt_id: entry for entry in rubric.entries}
    injected = provider is not None
    kind = runner_kind or ("injected" if injected else "fixture")
    if kind == "injected" and not injected:
        raise ValueError("injected runner requires a provider")
    answer_provider = provider or fixture_provider(rubric)
    answer_source = INJECTED_ANSWER_SOURCE if injected else PLACEHOLDER_ANSWER_SOURCE
    cells: list[ModelCell] = []
    for item in prompt_set.prompts:
        entry = rubric_by_prompt.get(item.prompt_id)
        for model_id in ids:
            if entry is None:
                cells.append(
                    ModelCell(
                        model_id=model_id,
                        prompt_id=item.prompt_id,
                        verdict="unrated",
                        answer_sha256=hashlib.sha256(b"").hexdigest(),
                        answer_chars=0,
                        latency_ms=0,
                        answer_source=UNRATED_ANSWER_SOURCE,
                    )
                )
                continue
            answer = answer_provider(model_id, item)
            decision = evaluate_completion(policy, entry.accepted, answer, truncated=False)
            text = answer or ""
            cells.append(
                ModelCell(
                    model_id=model_id,
                    prompt_id=item.prompt_id,
                    verdict=decision.status,
                    answer_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    answer_chars=len(text),
                    latency_ms=0,
                    answer_source=answer_source,
                )
            )
    rows = tuple(_row(model_id, tuple(cells)) for model_id in ids)
    return ModelCompareReport(
        prompt_set=prompt_set,
        model_ids=ids,
        rubric_id=rubric.rubric_id,
        rubric_digest=rubric.source_digest,
        policy_name=policy.id,
        cells=tuple(cells),
        rows=rows,
        runner_kind=kind,
        answer_source=answer_source,
    )


def _row(model_id: str, cells: tuple[ModelCell, ...]) -> ModelRow:
    model_cells = tuple(cell for cell in cells if cell.model_id == model_id)
    if not model_cells:
        raise ValueError("model produced no cells")
    counts: dict[str, int] = {verdict: 0 for verdict in VERDICTS}
    for cell in model_cells:
        counts[cell.verdict] += 1
    passed = counts["passed"]
    unrated = counts["unrated"]
    rated = len(model_cells) - unrated
    return ModelRow(
        model_id=model_id,
        prompt_count=len(model_cells),
        rated_count=rated,
        unrated_count=unrated,
        passed_count=passed,
        correct_rate=passed / rated if rated else 0.0,
        latency_ms_mean=sum(cell.latency_ms for cell in model_cells) / len(model_cells),
        verdict_counts={key: value for key, value in counts.items() if value},
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an offline multi-model output matrix (fixture providers)")
    parser.add_argument("--prompt-set", default=DEFAULT_PROMPT_SET, help="JSONL prompt set path")
    parser.add_argument("--rubric", default=DEFAULT_RUBRIC, help="correctness rubric JSON path")
    parser.add_argument("--models", default=",".join(DEFAULT_MODEL_IDS), help="comma-separated model ids")
    parser.add_argument("--json", dest="json_path", default=None, help="write the JSON report to this path")
    parser.add_argument("--markdown", dest="markdown_path", default=None, help="write the Markdown report to this path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prompt_set = load_prompt_set(args.prompt_set)
    rubric = load_compare_rubric(args.rubric)
    model_ids = tuple(item.strip() for item in args.models.split(",") if item.strip())
    report = run_model_compare(prompt_set, rubric, model_ids=model_ids)
    if args.json_path and args.json_path != "-":
        Path(args.json_path).write_text(
            json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if args.markdown_path and args.markdown_path != "-":
        Path(args.markdown_path).write_text(report.to_markdown(), encoding="utf-8")
    if args.json_path == "-":
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    if args.markdown_path == "-":
        print(report.to_markdown())
    if not args.json_path and not args.markdown_path:
        print(report.to_markdown())
    return 0


__all__ = [
    "DEFAULT_MODEL_IDS",
    "DEFAULT_PROMPT_SET",
    "DEFAULT_RUBRIC",
    "INJECTED_ANSWER_SOURCE",
    "MODEL_COMPARE_INPUT_SCHEMA",
    "MODEL_COMPARE_SCHEMA",
    "PLACEHOLDER_ANSWER_SOURCE",
    "UNRATED_ANSWER_SOURCE",
    "VERDICTS",
    "ComparePrompt",
    "ComparePromptSet",
    "ModelCell",
    "ModelCompareReport",
    "ModelRow",
    "fixture_provider",
    "load_compare_rubric",
    "load_prompt_set",
    "run_model_compare",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
