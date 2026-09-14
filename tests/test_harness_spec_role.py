import pytest

from harness_workbench.research import (
    SPEC_ROLE_SCHEMA,
    SpecProfileConfig,
    parse_prometheus_counters,
    run_spec_profile,
    run_spec_role_report,
    spec_delta,
)


_METRICS_TEMPLATE = """# HELP llamacpp:n_decode_total decodes
llamacpp:n_decode_total {decode}
# HELP llamacpp:spec_decode_num_draft_tokens_total draft tokens
# TYPE llamacpp:spec_decode_num_draft_tokens_total counter
llamacpp:spec_decode_num_draft_tokens_total {draft}
# TYPE llamacpp:spec_decode_num_accepted_tokens_total counter
llamacpp:spec_decode_num_accepted_tokens_total {accepted}
# TYPE llamacpp:spec_decode_num_drafts_total counter
llamacpp:spec_decode_num_drafts_total {drafts}
"""


class _FakeBackend:
    """Deterministic loopback stand-in with growing speculative counters."""

    def __init__(self, *, speculative: bool) -> None:
        self.speculative = speculative
        self.draft = 0
        self.accepted = 0
        self.drafts = 0

    def metrics(self, endpoint: str) -> str:
        return _METRICS_TEMPLATE.format(
            decode=64,
            draft=self.draft,
            accepted=self.accepted,
            drafts=self.drafts,
        )

    def chat(self, endpoint, payload, timeout):
        prompt = payload["messages"][-1]["content"]
        if self.speculative:
            self.draft += 12
            self.accepted += 9
            self.drafts += 4
        answer = f"answer::{prompt[:24]}"
        return {
            "choices": [{"message": {"role": "assistant", "content": answer}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": len(answer.split())},
            "timings": {"predicted_n": 24, "predicted_ms": 120.0},
        }


def test_prometheus_parser_and_spec_delta():
    counters = parse_prometheus_counters(
        _METRICS_TEMPLATE.format(decode=1, draft=10, accepted=7, drafts=3)
    )

    assert counters["llamacpp:spec_decode_num_draft_tokens_total"] == 10.0
    delta = spec_delta(
        {"llamacpp:spec_decode_num_draft_tokens_total": 10.0},
        {
            "llamacpp:spec_decode_num_draft_tokens_total": 22.0,
            "llamacpp:spec_decode_num_accepted_tokens_total": 9.0,
            "llamacpp:spec_decode_num_drafts_total": 4.0,
        },
    )
    assert delta == {"draft_tokens": 12, "accepted_tokens": 9, "drafts": 4}
    assert parse_prometheus_counters("# comment\n\nnonsense") == {}


def test_spec_profile_records_counters_and_throughput():
    backend = _FakeBackend(speculative=True)
    config = SpecProfileConfig(
        label="d4-draft-06b",
        target_model="Qwen3-4B",
        draft_model="Qwen3-0.6B",
        speculative=True,
    )

    results = run_spec_profile(
        config,
        endpoint="http://127.0.0.1:8082",
        prompts=(("p1", "hello"), ("p2", "world")),
        transport=backend.chat,
        metrics_reader=backend.metrics,
    )

    assert len(results) == 2
    first = results[0]
    assert first.draft_tokens == 12 and first.accepted_tokens == 9 and first.drafts == 4
    assert first.acceptance_rate == pytest.approx(0.75)
    assert first.tokens_per_round == pytest.approx(3.0)
    assert first.decode_tokens_per_s == pytest.approx(200.0)
    assert first.predicted_tokens == 24


def test_baseline_profile_rejects_speculative_counters():
    backend = _FakeBackend(speculative=False)
    config = SpecProfileConfig(label="d4-solo", target_model="Qwen3-4B")

    results = run_spec_profile(
        config,
        endpoint="http://127.0.0.1:8082",
        prompts=(("p1", "hello"),),
        transport=backend.chat,
        metrics_reader=backend.metrics,
    )

    assert results[0].draft_tokens == 0
    assert results[0].acceptance_rate == 0.0
    assert results[0].speculative is False


def test_role_report_marks_asym_checks_and_keeps_digest_stable():
    baseline_backend = _FakeBackend(speculative=False)
    spec_backend = _FakeBackend(speculative=True)
    configs = (
        SpecProfileConfig(label="d4-solo", target_model="Qwen3-4B"),
        SpecProfileConfig(
            label="d4-draft-06b",
            target_model="Qwen3-4B",
            draft_model="Qwen3-0.6B",
            speculative=True,
        ),
    )
    prompts = (("p1", "hello"), ("p2", "world"))

    def backend_for(config: SpecProfileConfig) -> _FakeBackend:
        return spec_backend if config.speculative else baseline_backend

    report = run_spec_role_report(
        configs,
        baseline_label="d4-solo",
        endpoint="http://127.0.0.1:8082",
        prompts=prompts,
        transport_factory=lambda config: backend_for(config).chat,
        metrics_reader_factory=lambda config: backend_for(config).metrics,
    )

    assert report.schema == SPEC_ROLE_SCHEMA
    rows = {summary.label: summary for summary in report.summaries}
    assert rows["d4-solo"].speedup_vs_baseline == pytest.approx(1.0)
    assert rows["d4-solo"].checks["asym_02_cost_gate"] is True
    assert rows["d4-draft-06b"].acceptance_rate == pytest.approx(0.75)
    assert rows["d4-draft-06b"].tokens_per_round == pytest.approx(3.0)
    assert rows["d4-draft-06b"].agreement_with_baseline == pytest.approx(1.0)
    assert rows["d4-draft-06b"].checks["asym_01_distribution_agreement"] is True
    assert rows["d4-draft-06b"].checks["asym_02_cost_gate"] is True
    assert "| `d4-draft-06b` |" in report.to_markdown()
    payload = report.as_dict()
    assert payload["evidence_scope"].startswith("loopback_speculative_role_profile")
    assert "answer_sha256" in payload["results"][0]
    assert all(key != "answer" for key in payload["results"][0])
    assert "predicted_ms" not in report.digest_payload()["results"][0]
    assert len(report.digest) == 64


def test_role_report_rejects_unknown_baseline_and_bad_configs():
    config = SpecProfileConfig(label="d4-solo", target_model="Qwen3-4B")
    with pytest.raises(ValueError):
        run_spec_role_report((config,), baseline_label="missing", endpoint="http://127.0.0.1:8082")
    with pytest.raises(ValueError):
        SpecProfileConfig(label="bad", target_model="Qwen3-4B", speculative=True)
    with pytest.raises(ValueError):
        SpecProfileConfig(label="bad", target_model="Qwen3-4B", draft_model="Qwen3-0.6B")
