import json
from pathlib import Path

import pytest

from harness_workbench.tools.model_compare import (
    DEFAULT_MODEL_IDS,
    MODEL_COMPARE_SCHEMA,
    PLACEHOLDER_ANSWER_SOURCE,
    UNRATED_ANSWER_SOURCE,
    ComparePrompt,
    load_compare_rubric,
    load_prompt_set,
    main,
    run_model_compare,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_SET_PATH = REPO_ROOT / "fixtures" / "prompt_sets" / "ps-qwen3-nothink" / "prompts.jsonl"
RUBRIC_PATH = REPO_ROOT / "fixtures" / "quality_rubrics" / "llm-objective-ps-v1-qwen3-nothink-v1.json"
PROMPT_COUNT = 30


def _fixtures():
    return load_prompt_set(PROMPT_SET_PATH), load_compare_rubric(RUBRIC_PATH)


def test_repo_fixtures_load_and_rubric_covers_a_judgeable_subset():
    prompt_set, rubric = _fixtures()
    prompt_ids = {item.prompt_id for item in prompt_set.prompts}
    rubric_ids = set(rubric.prompt_ids)

    assert prompt_set.prompt_set_id == "ps-qwen3-nothink"
    assert len(prompt_set.prompts) == PROMPT_COUNT
    assert prompt_set.source_ref == "prompts.jsonl"  # absolute input paths are redacted to the file name
    assert len(prompt_set.sha256) == 64
    # The correctness dimension only covers the objectively checkable subset.
    assert rubric_ids and rubric_ids <= prompt_ids
    assert len(rubric_ids) == 4


def test_fixture_matrix_is_reproducible_with_unrated_cells():
    prompt_set, rubric = _fixtures()
    rated_total = len(rubric.entries)
    first = run_model_compare(prompt_set, rubric)
    second = run_model_compare(prompt_set, rubric)

    assert first.schema == MODEL_COMPARE_SCHEMA
    assert first.runner_kind == "fixture"
    assert first.answer_source == PLACEHOLDER_ANSWER_SOURCE
    assert first.model_ids == DEFAULT_MODEL_IDS
    assert len(first.cells) == len(DEFAULT_MODEL_IDS) * PROMPT_COUNT
    assert first.digest == second.digest
    assert first.as_dict() == second.as_dict()
    payload = first.as_dict()
    assert payload["evidence_scope"] == "pipeline_and_judging_contract_only"
    assert payload["network_used"] is False
    assert payload["weights_loaded"] is False
    assert len(payload["unrated_prompt_ids"]) == PROMPT_COUNT - rated_total
    rated = [cell for cell in first.cells if cell.rated]
    assert len(rated) == len(DEFAULT_MODEL_IDS) * rated_total
    assert all(cell.verdict == "passed" for cell in rated)
    unrated = [cell for cell in first.cells if cell.verdict == "unrated"]
    assert unrated and all(cell.answer_source == UNRATED_ANSWER_SOURCE for cell in unrated)
    assert all(
        row.rated_count == rated_total and row.unrated_count == PROMPT_COUNT - rated_total
        for row in first.rows
    )
    assert all(row.correct_rate == 1.0 for row in first.rows)


def test_injected_provider_records_failures_and_missing_answers():
    prompt_set, rubric = _fixtures()
    rated_ids = tuple(rubric.prompt_ids)
    rated_total = len(rated_ids)
    accepted_by_prompt = {entry.prompt_id: entry.accepted[0] for entry in rubric.entries}

    def provider(model_id: str, prompt: ComparePrompt) -> str | None:
        if model_id == "QW1.8B":
            return None
        if model_id == "Qwen3-4B" and prompt.prompt_id == rated_ids[0]:
            return "完全无关的回答"
        return accepted_by_prompt.get(prompt.prompt_id, "placeholder")

    report = run_model_compare(prompt_set, rubric, provider=provider)
    rows = {row.model_id: row for row in report.rows}

    assert report.runner_kind == "injected"
    assert report.as_dict()["weights_loaded"] is True
    assert rows["QW1.8B"].verdict_counts.get("missing") == rated_total
    assert rows["QW1.8B"].correct_rate == 0.0
    assert rows["QW1.8B"].unrated_count == PROMPT_COUNT - rated_total
    assert rows["Qwen3-4B"].verdict_counts.get("failed") == 1
    assert rows["Qwen3-4B"].passed_count == rated_total - 1
    assert rows["Qwen3-4B"].rated_count == rated_total
    assert all(len(cell.answer_sha256) == 64 for cell in report.cells)
    assert "answer_text" not in report.cells[0].as_dict()


def test_matrix_shape_and_markdown_render_every_model():
    prompt_set, rubric = _fixtures()
    report = run_model_compare(prompt_set, rubric, model_ids=("Qwen3-4B", "QW1.8B"))
    matrix = report.matrix()
    markdown = report.to_markdown()

    assert set(matrix) == {item.prompt_id for item in prompt_set.prompts}
    assert all(set(row) == {"Qwen3-4B", "QW1.8B"} for row in matrix.values())
    assert "| prompt | `Qwen3-4B` | `QW1.8B` |" in markdown
    assert f"Rated prompts: {len(rubric.entries)} of {PROMPT_COUNT}" in markdown
    assert "| model | prompts | rated | passed | correct rate (rated) |" in markdown
    assert "Report digest:" in markdown
    assert len(report.matrix_digest) == 64


def test_report_rejects_invalid_inputs():
    prompt_set, rubric = _fixtures()
    with pytest.raises(ValueError):
        run_model_compare(prompt_set, rubric, model_ids=("Qwen3-4B", "Qwen3-4B"))
    with pytest.raises(ValueError):
        run_model_compare(prompt_set, rubric, model_ids=("bad model",))
    with pytest.raises(ValueError):
        run_model_compare(prompt_set, rubric, runner_kind="injected")
    with pytest.raises(ValueError):
        load_prompt_set(REPO_ROOT / "fixtures" / "prompt_sets" / "missing.jsonl")
    with pytest.raises(ValueError):
        load_prompt_set(Path("C:/absolute/path.jsonl"))


def test_cli_writes_json_and_markdown_artifacts(tmp_path):
    json_path = tmp_path / "matrix.json"
    markdown_path = tmp_path / "matrix.md"

    code = main(
        [
            "--prompt-set",
            str(PROMPT_SET_PATH),
            "--rubric",
            str(RUBRIC_PATH),
            "--models",
            "Qwen3-4B,QW1.8B",
            "--json",
            str(json_path),
            "--markdown",
            str(markdown_path),
        ]
    )

    assert code == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema"] == MODEL_COMPARE_SCHEMA
    assert payload["model_ids"] == ["Qwen3-4B", "QW1.8B"]
    assert len(payload["report_digest"]) == 64
    assert "TOOL-MODEL-CMP-01" in markdown_path.read_text(encoding="utf-8")
