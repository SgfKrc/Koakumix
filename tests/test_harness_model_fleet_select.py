"""SIDE-KOAKU-03 tests: vision/text fleet selection with explainable degradation.

The harness must pick a vision or text model from the fleet for the device that is
actually present, and when the vision capability is missing it must degrade in a
readable way instead of loading a model the device cannot run.
"""

from __future__ import annotations

import pytest

from harness_workbench.model_profiles import (
    CAPABILITY_NAMES,
    SELECTION_SCHEMA,
    CapabilityState,
    DeviceProfile,
    FleetSelector,
    ModelProfile,
    ProfileValidationError,
    select_fleet_model,
)

_EVIDENCE = {"artifact_digest_mode": "full_stream", "fixture_set": "small-model-core-v1"}


def _caps(multimodal: str = "unknown") -> dict[str, CapabilityState]:
    capabilities = {
        name: CapabilityState("unknown", ())
        for name in CAPABILITY_NAMES
    }
    capabilities["multimodal"] = CapabilityState(multimodal, ("fixture_v1",))
    return capabilities


def _profile(
    model_id: str,
    *,
    multimodal: str = "unknown",
    status: str = "candidate",
    min_vram_bytes: int | None = None,
    revision: str = "fixture-v1",
    artifact: bool = True,
) -> ModelProfile:
    profile = ModelProfile(
        model_id=model_id,
        revision=revision,
        backend="llama_server",
        format="gguf",
        artifact_sha256=("a" * 64) if artifact else None,
        context={"n_ctx": 4096, "input_budget": 2048, "max_new_tokens": 256},
        resources={"min_vram_bytes": min_vram_bytes} if min_vram_bytes is not None else {},
        capabilities=_caps(multimodal),
        status=status,
        evidence=dict(_EVIDENCE),
    )
    return profile


CUDA_8G = DeviceProfile(platform="pc", accelerator="cuda", memory_bytes=16 * 1024**3, vram_bytes=8188 * 1024**2)
ANDROID = DeviceProfile(platform="android", accelerator="cpu")
NO_MEASUREMENT = DeviceProfile(platform="pc")


def test_vision_role_picks_a_verified_multimodal_model() -> None:
    # Vision needs a *verified* profile AND a verified multimodal capability.
    text = _profile("qwen/text", multimodal="unknown", status="verified")
    vision = _profile("qwen/vision", multimodal="verified", status="verified")
    selection = FleetSelector((text, vision)).select(role="vision", device=CUDA_8G)

    assert selection.available is True
    assert selection.mode == "multimodal"
    assert selection.selected_profile_id == vision.profile_id
    assert selection.degradations == ()
    assert selection.as_dict()["selection_schema"] == SELECTION_SCHEMA


def test_declared_multimodal_is_not_enough_for_the_vision_role() -> None:
    # probe.py only ever reaches "declared"; that must not unlock vision mode.
    declared = _profile("qwen/vl-declared", multimodal="declared", status="verified")
    selection = FleetSelector((declared,)).select(role="vision", device=CUDA_8G)

    assert selection.mode != "multimodal"
    assert any("mode_not_allowed:multimodal" in reason for reason in selection.reasons)
    assert "image_understanding_disabled" in selection.degradations


def test_vision_role_degrades_to_text_with_a_readable_reason() -> None:
    text = _profile("qw1/text-only", multimodal="unknown")
    selection = FleetSelector((text,)).select(role="vision", device=ANDROID)

    # Degraded, not failed: a text model is chosen and the loss is stated.
    assert selection.available is True
    assert selection.mode == "degraded_text_only"
    assert selection.selected_profile_id == text.profile_id
    assert selection.degradations == ("image_understanding_disabled",)
    assert any("mode_not_allowed:multimodal" in reason for reason in selection.reasons)


def test_vision_role_reports_unavailable_when_no_text_model_either() -> None:
    rejected = _profile("dead/model", status="rejected")
    selection = FleetSelector((rejected,)).select(role="vision", device=ANDROID)

    assert selection.available is False
    assert selection.mode == "unavailable"
    assert selection.profile is None
    assert selection.degradations == ("image_understanding_disabled", "no_model_available")
    assert "no_text_model_either" in selection.reasons


def test_declared_vram_requirement_blocks_without_an_accelerator() -> None:
    big = _profile("qwen/big-vision", multimodal="verified", min_vram_bytes=8 * 1024**3)
    selection = FleetSelector((big,)).select(role="vision", device=ANDROID)

    assert selection.available is False
    candidate = selection.candidates[0]
    assert candidate.admissible is False
    assert candidate.resource_status == "blocked"
    assert any(reason.startswith("insufficient_vram:") for reason in candidate.reasons)


def test_declared_vram_requirement_blocks_when_the_card_is_too_small() -> None:
    big = _profile("qwen/big-vision", multimodal="verified", min_vram_bytes=16 * 1024**3)
    selection = FleetSelector((big,)).select(role="vision", device=CUDA_8G)

    assert selection.available is False
    assert selection.candidates[0].resource_status == "blocked"


def test_undeclared_resource_requirement_is_reported_but_never_invented() -> None:
    text = _profile("qw1/text-only")
    selection = FleetSelector((text,)).select(role="answer", device=ANDROID)

    assert selection.available is True
    candidate = selection.candidates[0]
    assert candidate.resource_status == "unknown"
    assert "resource_requirement_undeclared" in candidate.reasons


def test_selection_prefers_verified_and_is_deterministic() -> None:
    unknown = _profile("a/unknown", status="unknown")
    verified = _profile("b/verified", status="verified")
    fleet = (unknown, verified)

    first = FleetSelector(fleet).select(role="answer", device=CUDA_8G)
    second = FleetSelector(tuple(reversed(fleet))).select(role="answer", device=CUDA_8G)

    assert first.selected_profile_id == verified.profile_id
    assert first.selected_profile_id == second.selected_profile_id
    assert [candidate.profile_id for candidate in first.candidates] == sorted(
        candidate.profile_id for candidate in first.candidates
    ) or first.candidates[0].profile_id == verified.profile_id


def test_unbound_profile_sorts_after_a_bound_one_with_the_same_status() -> None:
    unbound = _profile("z/unbound", revision="fixture-v1", artifact=False)
    bound = _profile("z/bound", revision="fixture-v1", artifact=True)
    selection = FleetSelector((unbound, bound)).select(role="answer", device=NO_MEASUREMENT)

    assert selection.available is True
    assert selection.selected_profile_id == bound.profile_id
    assert selection.candidates[0].artifact_bound is True


def test_select_fleet_model_convenience_wrapper_matches_selector() -> None:
    text = _profile("qw1/text-only")
    wrapped = select_fleet_model((text,), role="answer", device=CUDA_8G)
    direct = FleetSelector((text,)).select(role="answer", device=CUDA_8G)

    assert wrapped.selected_profile_id == direct.selected_profile_id
    assert wrapped.as_dict()["device"]["has_accelerator"] is True


def test_bad_role_and_bad_device_are_rejected() -> None:
    text = _profile("qw1/text-only")

    with pytest.raises(ProfileValidationError, match="selection role"):
        FleetSelector((text,)).select(role="embedding", device=CUDA_8G)
    with pytest.raises(ProfileValidationError, match="platform"):
        DeviceProfile(platform="toaster")
    with pytest.raises(ProfileValidationError, match="accelerator"):
        DeviceProfile(accelerator="magic")
    with pytest.raises(ProfileValidationError, match="vram_bytes"):
        DeviceProfile(vram_bytes=-1)


def test_device_profile_round_trips_and_keeps_unknowns_unknown() -> None:
    assert DeviceProfile.from_dict({"platform": "pc", "accelerator": "cpu"}) == DeviceProfile(
        platform="pc", accelerator="cpu"
    )
    assert NO_MEASUREMENT.has_accelerator is False
    assert DeviceProfile().as_dict() == {
        "platform": "unknown",
        "accelerator": "unknown",
        "memory_bytes": None,
        "vram_bytes": None,
        "has_accelerator": False,
    }
