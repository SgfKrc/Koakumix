import pytest

from harness_workbench.research import (
    ABLATION_SCHEMA,
    ARMS,
    build_harness_ablation_fixture,
    run_harness_ablation,
)


def test_ablation_fixture_is_stable_and_covers_every_task_kind():
    first = build_harness_ablation_fixture()
    second = build_harness_ablation_fixture()

    assert first.rounds == 3
    assert len(first.tasks) == 4
    assert {task.kind for task in first.tasks} == {
        "over_budget_multi_turn",
        "cross_session_memory",
        "rag_citation",
    }
    assert first.digest == second.digest
    assert first.as_dict()["fixture_digest"] == first.digest


def test_ablation_runs_three_arms_without_model_or_network():
    report = run_harness_ablation()

    assert report.schema == ABLATION_SCHEMA
    assert len(report.observations) == len(ARMS) * 4 * 3
    assert {item.arm for item in report.observations} == set(ARMS)
    assert all(item.runner_kind == "fixture" for item in report.observations)
    assert all(not item.network_used and not item.weights_loaded for item in report.observations)
    assert {summary.arm for summary in report.summaries} == set(ARMS)


def test_ablation_replay_is_digest_stable_and_round_stable():
    first = run_harness_ablation()
    second = run_harness_ablation()

    assert first.digest == second.digest
    assert first.as_dict() == second.as_dict()
    assert all(summary.stable and summary.stability_margin == 0.0 for summary in first.summaries)
    assert all(len(first.series()[arm]) == 12 for arm in ARMS)


def test_ablation_arms_separate_on_the_fixed_task_set():
    report = run_harness_ablation()
    rows = {summary.arm: summary for summary in report.summaries}

    baseline = rows["baseline"]
    assert baseline.task_success_rate == 0.0
    assert baseline.memory_recall_rate == 0.0
    assert baseline.citation_accuracy == 0.0
    assert baseline.token_cost_ratio == 1.0

    retrieval = rows["rag_memory"]
    assert retrieval.citation_accuracy == 1.0
    assert retrieval.citation_attempts == 3
    assert retrieval.memory_recall_rate == 1.0
    assert retrieval.task_success_rate > baseline.task_success_rate
    assert retrieval.token_cost_ratio >= 1.0

    compress = rows["compress_roles"]
    assert compress.citation_accuracy == 0.0
    assert compress.citation_attempts == 3
    assert compress.memory_recall_rate == 1.0
    assert compress.task_success_rate > baseline.task_success_rate


def test_ablation_cells_expose_per_task_outcomes_and_markdown():
    report = run_harness_ablation()
    cells = {(item.arm, item.task_id): item for item in report.observations}
    markdown = report.to_markdown()

    assert cells["baseline", "cross-session-units"].memory_recall_hits == 0
    assert cells["rag_memory", "cross-session-units"].memory_recall_hits == 1
    assert cells["rag_memory", "rag-budget-omissions"].citation_correct is True
    assert cells["baseline", "rag-budget-omissions"].citation_correct is False
    assert "| `baseline` |" in markdown
    assert "Real-run handoff" in markdown
    assert report.as_dict()["evidence_scope"] == "pipeline_and_metric_contract_only"


def test_ablation_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        run_harness_ablation(input_budget=0)
    with pytest.raises(ValueError):
        run_harness_ablation(seed=18, fixture=build_harness_ablation_fixture(seed=17))
