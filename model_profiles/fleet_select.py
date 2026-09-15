"""Pick a vision or text model from the fleet for the device that is actually present.

SIDE-KOAKU-03: the harness must be able to choose a vision/text model from the
model fleet, and when the required capability is missing it must degrade in a way
a user can read -- never by silently loading a bigger model than the device can
run.

Design boundaries:

* the device is **injected**, not probed here: the caller (main-repo device
  profiler, Android-side probe, fixture) supplies a :class:`DeviceProfile`, so the
  selection stays deterministic and testable offline;
* capability admission is delegated to :class:`CapabilityGate`, so an unknown or
  merely ``declared`` multimodal capability never becomes a vision model;
* a resource requirement only blocks when the profile *declares* one
  (``resources.min_vram_bytes`` / ``min_memory_bytes``); an undeclared
  requirement is reported as unknown instead of being invented;
* every decision carries ``reasons`` and ``degradations`` so "no vision model on
  this device" is explainable rather than silent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .capability_gate import CapabilityGate, GateDecision
from .schema import ModelProfile, ProfileValidationError


SELECTION_SCHEMA = "qlh.harness.fleet_selection.v1"
SELECTION_ROLES = ("answer", "vision")
PLATFORMS = ("pc", "android", "server", "unknown")
ACCELERATORS = ("cuda", "metal", "cpu", "unknown")
_REQUIRED_MODE = {"answer": "answer", "vision": "multimodal"}
_STATUS_RANK = {"verified": 3, "candidate": 2, "unknown": 1, "rejected": 0}


@dataclass(frozen=True, slots=True)
class DeviceProfile:
    """The device the caller wants to run on; all fields are optional.

    ``vram_bytes``/``memory_bytes`` may be ``None`` when the caller has no
    measurement; that is reported as unknown rather than assumed to be enough.
    """

    platform: str = "unknown"
    accelerator: str = "unknown"
    memory_bytes: int | None = None
    vram_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.platform not in PLATFORMS:
            raise ProfileValidationError(f"unsupported device platform: {self.platform}")
        if self.accelerator not in ACCELERATORS:
            raise ProfileValidationError(f"unsupported device accelerator: {self.accelerator}")
        for name in ("memory_bytes", "vram_bytes"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ProfileValidationError(f"{name} must be a non-negative integer or null")

    @property
    def has_accelerator(self) -> bool:
        return self.accelerator in {"cuda", "metal"} and bool(self.vram_bytes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "accelerator": self.accelerator,
            "memory_bytes": self.memory_bytes,
            "vram_bytes": self.vram_bytes,
            "has_accelerator": self.has_accelerator,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DeviceProfile":
        if not isinstance(value, Mapping):
            raise ProfileValidationError("device profile must be an object")
        return cls(
            platform=str(value.get("platform", "unknown")),
            accelerator=str(value.get("accelerator", "unknown")),
            memory_bytes=value.get("memory_bytes"),
            vram_bytes=value.get("vram_bytes"),
        )


def _resource_check(profile: ModelProfile, device: DeviceProfile) -> tuple[str, tuple[str, ...]]:
    """Return ``("ok"|"blocked"|"unknown", reasons)`` for declared requirements only."""

    resources = profile.resources or {}
    min_vram = resources.get("min_vram_bytes")
    min_memory = resources.get("min_memory_bytes")
    declared = False

    if isinstance(min_vram, int) and not isinstance(min_vram, bool):
        declared = True
        if not device.has_accelerator:
            return "blocked", (f"insufficient_vram:{min_vram}>none",)
        if isinstance(device.vram_bytes, int) and device.vram_bytes < min_vram:
            return "blocked", (f"insufficient_vram:{min_vram}>{device.vram_bytes}",)
    if isinstance(min_memory, int) and not isinstance(min_memory, bool):
        declared = True
        if isinstance(device.memory_bytes, int) and device.memory_bytes < min_memory:
            return "blocked", (f"insufficient_memory:{min_memory}>{device.memory_bytes}",)

    if not declared:
        return "unknown", ("resource_requirement_undeclared",)
    return "ok", ()


@dataclass(frozen=True, slots=True)
class FleetCandidate:
    """One evaluated fleet member: gate decision plus why it cannot serve."""

    profile_id: str
    model_id: str
    format: str
    backend: str
    status: str
    resource_status: str
    artifact_bound: bool
    admissible: bool
    reasons: tuple[str, ...]
    gate: GateDecision

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "model_id": self.model_id,
            "format": self.format,
            "backend": self.backend,
            "status": self.status,
            "resource_status": self.resource_status,
            "artifact_bound": self.artifact_bound,
            "admissible": self.admissible,
            "reasons": list(self.reasons),
            "gate": self.gate.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class FleetSelection:
    """The selected model plus an explainable degradation trail."""

    role: str
    available: bool
    mode: str
    selected_profile_id: str | None
    profile: ModelProfile | None
    device: DeviceProfile
    reasons: tuple[str, ...]
    degradations: tuple[str, ...]
    candidates: tuple[FleetCandidate, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "selection_schema": SELECTION_SCHEMA,
            "role": self.role,
            "available": self.available,
            "mode": self.mode,
            "selected_profile_id": self.selected_profile_id,
            "device": self.device.as_dict(),
            "reasons": list(self.reasons),
            "degradations": list(self.degradations),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


def _sort_key(candidate: FleetCandidate) -> tuple[int, int, str]:
    # Strongest gate status first, then profiles bound to a real artifact
    # (artifact_sha256 present), then a stable id order for determinism.
    return (
        -_STATUS_RANK.get(candidate.status, 0),
        0 if candidate.artifact_bound else 1,
        candidate.profile_id,
    )


class FleetSelector:
    """Choose one fleet member for one role on one device, fail-closed."""

    def __init__(self, profiles: Iterable[ModelProfile]):
        self.profiles: tuple[ModelProfile, ...] = tuple(profiles)
        self._gate = CapabilityGate()

    def _evaluate(self, profile: ModelProfile, role: str, device: DeviceProfile) -> FleetCandidate:
        gate = self._gate.evaluate(profile, role=role)
        resource_status, resource_reasons = _resource_check(profile, device)
        required_mode = _REQUIRED_MODE[role]
        reasons = list(resource_reasons)
        if not gate.can(required_mode):
            reasons.append(f"mode_not_allowed:{required_mode}")
        return FleetCandidate(
            profile_id=profile.profile_id,
            model_id=profile.model_id,
            format=profile.format,
            backend=profile.backend,
            status=gate.status,
            resource_status=resource_status,
            artifact_bound=profile.artifact_sha256 is not None,
            admissible=resource_status != "blocked" and gate.can(required_mode),
            reasons=tuple(dict.fromkeys(reasons)),
            gate=gate,
        )

    def select(self, *, role: str = "answer", device: DeviceProfile | None = None) -> FleetSelection:
        if role not in SELECTION_ROLES:
            raise ProfileValidationError(f"unsupported selection role: {role}")
        resolved = device or DeviceProfile()
        candidates = tuple(
            sorted(
                (self._evaluate(profile, role, resolved) for profile in self.profiles),
                key=_sort_key,
            )
        )
        by_id = {profile.profile_id: profile for profile in self.profiles}
        admissible = [candidate for candidate in candidates if candidate.admissible]

        if admissible:
            chosen = admissible[0]
            return FleetSelection(
                role=role,
                available=True,
                mode=_REQUIRED_MODE[role],
                selected_profile_id=chosen.profile_id,
                profile=by_id[chosen.profile_id],
                device=resolved,
                reasons=(),
                degradations=(),
                candidates=candidates,
            )

        reasons = tuple(
            dict.fromkeys(
                f"{candidate.profile_id}:{reason}"
                for candidate in candidates
                for reason in candidate.reasons
            )
        ) or ("fleet_empty",)

        if role == "vision":
            # Degrade to a text model instead of loading a vision model the
            # device cannot run; say plainly that image understanding is off.
            text = self.select(role="answer", device=resolved)
            if text.available:
                return FleetSelection(
                    role="vision",
                    available=True,
                    mode="degraded_text_only",
                    selected_profile_id=text.selected_profile_id,
                    profile=text.profile,
                    device=resolved,
                    reasons=reasons,
                    degradations=("image_understanding_disabled",),
                    candidates=candidates,
                )
            return FleetSelection(
                role="vision",
                available=False,
                mode="unavailable",
                selected_profile_id=None,
                profile=None,
                device=resolved,
                reasons=reasons + ("no_text_model_either",),
                degradations=("image_understanding_disabled", "no_model_available"),
                candidates=candidates,
            )

        return FleetSelection(
            role=role,
            available=False,
            mode="unavailable",
            selected_profile_id=None,
            profile=None,
            device=resolved,
            reasons=reasons,
            degradations=("no_model_available",),
            candidates=candidates,
        )


def select_fleet_model(
    profiles: Iterable[ModelProfile],
    *,
    role: str = "answer",
    device: DeviceProfile | None = None,
) -> FleetSelection:
    """Convenience wrapper around :class:`FleetSelector`."""

    return FleetSelector(profiles).select(role=role, device=device)
