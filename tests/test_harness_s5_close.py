import time

import pytest

from harness_workbench.research import (
    S5_CLOSE_SCHEMA,
    EvidenceEntry,
    OllamaComparisonConfig,
    build_pareto_section,
    run_ollama_comparison,
    run_s5_close,
)


def _fake_ollama(endpoint, payload, timeout):
    prompt = payload["messages"][-1]["content"]
    answer = f"ollama::{prompt[:20]}"
    time.sleep(0.002)  # keep injected latency measurable above ms resolution
    return {
        "choices": [{"message": {"role": "assistant", "content": answer}}],
        "usage": {"prompt_tokens": 18, "completion_tokens": 12},
    }


def _fake_baseline(endpoint, payload, timeout):
    prompt = payload["messages"][-1]["content"]
    answer = f"ollama::{prompt[:20]}" if "capital" in prompt else f"llama::{prompt[:20]}"
    time.sleep(0.001)
    return {
        "choices": [{"message": {"role": "assistant", "content": answer}}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10},
    }


def test_closeout_offline_report_is_reproducible():
    first = run_s5_close()
    second = run_s5_close()

    assert first.schema == S5_CLOSE_SCHEMA
    assert first.digest == second.digest
    assert first.as_dict() == second.as_dict()
    assert first.contract_drift.case_count == 5
    assert first.contract_drift.matched == 4
    assert first.contract_drift.drifted == 1
    assert first.contract_drift.failed == 0
    assert first.contract_drift.network_used is False
    assert first.ollama.status == "not_run"
    assert first.ollama.reason == "transports_not_injected"
    assert first.pareto.scope == "fixture"
    assert len(first.pareto.points) == 4
    assert len(first.ledger) >= 6
    assert first.as_dict()["weights_loaded"] is False


def test_contract_drift_section_names_the_drifting_case():
    report = run_s5_close()

    assert report.contract_drift.valid is True
    assert report.contract_drift.drift_cases
    case_id, mismatch = report.contract_drift.drift_cases[0]
    assert case_id == "chat-invalid"
    assert "status" in mismatch or "shape" in mismatch or "error" in mismatch


def test_ollama_comparison_requires_transports_and_measures_agreement():
    config = OllamaComparisonConfig(prompts=(("p1", "Name the capital of France."), ("p2", "Explain caching.")))
    not_run = run_ollama_comparison(config)

    assert not_run.status == "not_run" and not not_run.entries

    completed = run_ollama_comparison(
        config,
        ollama_transport=_fake_ollama,
        baseline_transport=_fake_baseline,
    )

    assert completed.status == "completed"
    assert len(completed.entries) == 2
    assert completed.agreement_rate == pytest.approx(0.5)
    assert completed.latency_ratio > 0
    assert completed.entries[0].agreement is True
    assert completed.entries[1].agreement is False


def test_closeout_report_uses_injected_comparison_and_keeps_digest_free_of_latency():
    config = OllamaComparisonConfig(prompts=(("p1", "Name the capital of France."),))
    report = run_s5_close(
        ollama_config=config,
        ollama_transport=_fake_ollama,
        baseline_transport=_fake_baseline,
    )

    payload = report.as_dict()
    assert payload["weights_loaded"] is True
    assert payload["ollama_comparison"]["status"] == "completed"
    assert "latency_ms" not in report.digest_payload()["ollama"]["entries"][0]
    assert all("answer" not in key for key in payload["contract_drift"])
    assert "Ollama `qwen3-vl:4b`" in report.to_markdown()


def test_pareto_section_marks_dominated_points_and_keeps_scope():
    rows = (
        {"id": "big", "quality_rate": 0.4, "latency_p95_ms": 3000.0, "rss_peak_bytes": 3_000_000_000},
        {"id": "small", "quality_rate": 0.2, "latency_p95_ms": 800.0, "rss_peak_bytes": 1_000_000_000},
    )
    section = build_pareto_section(rows)

    assert section.scope == "live_loopback"
    assert section.dominated_count == 0
    dominated = build_pareto_section(
        rows
        + (
            {"id": "slow", "quality_rate": 0.1, "latency_p95_ms": 4000.0, "rss_peak_bytes": 4_000_000_000},
        )
    )
    assert dominated.dominated_count == 1
    with pytest.raises(ValueError):
        build_pareto_section(rows, scope="unknown-scope")


def test_evidence_ledger_enforces_boundaries_and_relative_paths():
    with pytest.raises(ValueError):
        EvidenceEntry(
            ticket="T-1",
            theme="theme",
            status="completed",
            key_numbers="none",
            evidence_paths=(r"G:\abs\path.json",),
            scope="fixture",
            boundary="boundary",
        )
    with pytest.raises(ValueError):
        EvidenceEntry(
            ticket="T-1",
            theme="theme",
            status="completed",
            key_numbers="none",
            evidence_paths=("build/x.json",),
            scope="fixture",
            boundary="   ",
        )
    with pytest.raises(ValueError):
        EvidenceEntry(
            ticket="T-1",
            theme="theme",
            status="completed",
            key_numbers="none",
            evidence_paths=("build/../secret.json",),
            scope="fixture",
            boundary="boundary",
        )


def test_closeout_markdown_lists_all_sections():
    markdown = run_s5_close().to_markdown()

    assert "S5-CLOSE-01 Koakumix closeout report" in markdown
    assert "## Contract drift" in markdown
    assert "## Ollama comparison" in markdown
    assert "## Pareto marking" in markdown
    assert "## Evidence ledger" in markdown
    assert "| `EX-HARNESS-01` |" in markdown
    assert "| `EX-SPEC-ROLE-01` |" in markdown
