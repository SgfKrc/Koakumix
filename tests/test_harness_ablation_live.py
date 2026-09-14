import pytest

from harness_workbench.research import (
    ARMS,
    LIVE_ABLATION_SCHEMA,
    LocalBackendError,
    assert_loopback_endpoint,
    judge_answer,
    run_live_ablation,
)


def _fake_transport(endpoint, payload, timeout):
    """Simulate a model that can only answer from the supplied context."""

    text = payload["messages"][-1]["content"]
    if "INDIGO-LANTERN" in text:
        answer = "The staging gate codename is INDIGO-LANTERN."
    elif "internal build numbers" in text:
        answer = "The constraint is never to reveal internal build numbers."
    elif "metric units" in text:
        answer = "You asked for metric units."
    elif "silent truncation" in text:
        answer = "Omission counters must be reported instead of silent truncation."
    else:
        answer = "I do not know."
    return {
        "choices": [{"message": {"role": "assistant", "content": answer}}],
        "usage": {"prompt_tokens": 128, "completion_tokens": 12},
    }


@pytest.fixture(scope="module")
def live_report():
    return run_live_ablation(transport=_fake_transport)


def test_live_ablation_keeps_offline_arm_separation(live_report):
    rows = {summary.arm: summary for summary in live_report.summaries}

    assert live_report.schema == LIVE_ABLATION_SCHEMA
    assert len(live_report.observations) == len(ARMS) * 4 * 3
    assert rows["baseline"].task_success_rate == 0.0
    assert rows["rag_memory"].task_success_rate == 1.0
    assert rows["compress_roles"].task_success_rate == 0.75
    assert all(summary.stable for summary in live_report.summaries)
    assert rows["rag_memory"].token_cost_ratio > 1.0
    assert rows["compress_roles"].token_cost_ratio > rows["rag_memory"].token_cost_ratio


def test_live_report_digest_is_latency_free_and_answers_are_hashed(live_report):
    second = run_live_ablation(transport=_fake_transport)
    payload = live_report.as_dict()

    assert live_report.digest == second.digest
    assert payload["report_digest"] == live_report.digest
    assert payload["runner_kind"] == "llama_server"
    assert payload["weights_loaded"] is True
    assert payload["network_used"] is False
    assert payload["evidence_scope"] == "live_loopback_inference_with_loose_contains_judge"
    assert all(len(item["answer_sha256"]) == 64 for item in payload["observations"])
    assert all(
        set(item)
        == {
            "arm",
            "task_id",
            "task_kind",
            "round_index",
            "judge_phrase",
            "judge_mode",
            "task_success",
            "answer_sha256",
            "answer_chars",
            "input_tokens",
            "prompt_tokens",
            "completion_tokens",
            "latency_ms",
            "fixture_task_success",
            "memory_recall_rate",
            "citation_expected",
            "citation_correct",
            "model_id",
            "endpoint_kind",
            "runner_kind",
            "weights_loaded",
            "network_used",
            "notices",
        }
        for item in payload["observations"]
    )
    assert all(item["answer_chars"] > 0 for item in payload["observations"])
    assert "latency" not in live_report.digest_payload()["cells"][0]


def test_live_transport_rejects_non_loopback_endpoints():
    assert assert_loopback_endpoint("http://127.0.0.1:8081/") == "http://127.0.0.1:8081"
    assert assert_loopback_endpoint("http://localhost:8081") == "http://localhost:8081"
    for bad in (
        "http://192.168.1.10:8081",
        "https://127.0.0.1:8081",
        "http://user:pass@127.0.0.1:8081",
        "http://127.0.0.1:8081/?x=1",
    ):
        with pytest.raises(ValueError):
            assert_loopback_endpoint(bad)
    with pytest.raises(ValueError):
        run_live_ablation(endpoint="http://10.0.0.5:8081", transport=_fake_transport)


def test_live_judge_is_case_and_separator_insensitive():
    assert judge_answer("The codename is Indigo-Lantern.", "over-budget-early-fact") is True
    assert judge_answer("Indigo Lantern", "over-budget-early-fact") is True
    assert judge_answer("I do not know.", "over-budget-early-fact") is False
    with pytest.raises(ValueError):
        judge_answer("anything", "unknown-task")


def test_live_backend_failures_propagate_without_silent_success():
    def broken_transport(endpoint, payload, timeout):
        raise LocalBackendError("backend_unreachable")

    with pytest.raises(LocalBackendError):
        run_live_ablation(transport=broken_transport)


def test_live_report_markdown_documents_scope(live_report):
    markdown = live_report.to_markdown()

    assert "live ablation (loopback llama-server)" in markdown
    assert "cost ratio" in markdown
    assert "external network used: `false`" in markdown
    assert "| `baseline` |" in markdown
