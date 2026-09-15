"""Edit contracts for advanced image editing: img2img, inpaint, IP-Adapter, instruct.

S3.2-EDIT-01: the four editing paths need a backend-neutral request contract, a
remote mapping shape, and an honest statement of where each path can actually run
today.  The local executor is known-unavailable (no CUDA/diffusers in this ticket)
and no remote editing endpoint has been verified, so every path is reported as a
contract with ``contract_only`` runtime status rather than a working feature.

Design boundaries:

* source/mask/reference images are :class:`ImageAssetRef` values -- identifiers and
  digests only.  No bytes, no base64 and no local path ever enter a request or its
  remote payload;
* required components are declared per mode (inpaint needs a mask, IP-Adapter needs
  a reference, img2img/instruct need a source) and enforced fail-closed;
* :func:`resolve_edit_availability` reuses the adapter's advertised capabilities, so
  a path is only called available when an adapter actually claims it *and* reports a
  live runtime; otherwise the reason is explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .contracts import ImageAdapterCapabilities, ImageRequestError
from .refs import ImageAssetRef, ImageRefError


EDIT_REQUEST_SCHEMA = "qlh.harness.image_edit_request.v1"
EDIT_MODES = ("img2img", "inpaint", "ip_adapter", "instruct")
CONTRACT_ONLY = "contract_only"
UNAVAILABLE = "unavailable"

MAX_INSTRUCTION_CHARS = 1000
MAX_STRENGTH = 1.0


@dataclass(frozen=True, slots=True)
class EditModeSpec:
    """What one editing mode requires and where it can actually run."""

    mode: str
    requires: tuple[str, ...]
    optional: tuple[str, ...]
    local_status: str
    remote_status: str
    notes: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "requires": list(self.requires),
            "optional": list(self.optional),
            "local_status": self.local_status,
            "remote_status": self.remote_status,
            "notes": self.notes,
        }


EDIT_MODE_SPECS: Mapping[str, EditModeSpec] = {
    "img2img": EditModeSpec(
        mode="img2img",
        requires=("source",),
        optional=("mask", "reference"),
        local_status=UNAVAILABLE,
        remote_status=CONTRACT_ONLY,
        notes="denoise the source at `strength`; local diffusers executor not wired",
    ),
    "inpaint": EditModeSpec(
        mode="inpaint",
        requires=("source", "mask"),
        optional=("reference",),
        local_status=UNAVAILABLE,
        remote_status=CONTRACT_ONLY,
        notes="regenerate the masked region; requires a mask reference",
    ),
    "ip_adapter": EditModeSpec(
        mode="ip_adapter",
        requires=("reference",),
        optional=("source", "mask"),
        local_status=UNAVAILABLE,
        remote_status=CONTRACT_ONLY,
        notes="style/content transfer from a reference image; needs IP-Adapter weights",
    ),
    "instruct": EditModeSpec(
        mode="instruct",
        requires=("source", "instruction"),
        optional=("reference",),
        local_status=UNAVAILABLE,
        remote_status=CONTRACT_ONLY,
        notes="instruction-driven edit (e.g. InstructPix2Pix); no weights or runtime here",
    ),
}


def edit_mode_spec(mode: str) -> EditModeSpec:
    if mode not in EDIT_MODE_SPECS:
        raise ImageRequestError(
            "mode must be one of: " + ", ".join(EDIT_MODES), code="invalid_edit_mode"
        )
    return EDIT_MODE_SPECS[mode]


@dataclass(frozen=True, slots=True)
class ImageEditRequest:
    """A backend-neutral editing request; images are referenced, never embedded."""

    mode: str
    prompt: str = ""
    source: ImageAssetRef | None = None
    mask: ImageAssetRef | None = None
    reference: ImageAssetRef | None = None
    instruction: str = ""
    negative_prompt: str = ""
    strength: float = 0.6
    width: int = 512
    height: int = 512
    steps: int = 28
    guidance_scale: float = 7.5
    seed: int | None = None
    model: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    @property
    def spec(self) -> EditModeSpec:
        return edit_mode_spec(self.mode)

    def components(self) -> dict[str, Any]:
        return {"source": self.source, "mask": self.mask, "reference": self.reference}

    def validate(self) -> None:
        spec = edit_mode_spec(self.mode)
        for name in ("source", "mask", "reference"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, ImageAssetRef):
                raise ImageRequestError(
                    f"{name} must be an ImageAssetRef", code="invalid_edit_reference"
                )
        components = self.components()
        missing = [
            name for name in spec.requires
            if name in components and components[name] is None
        ]
        if "instruction" in spec.requires and not self.instruction.strip():
            missing.append("instruction")
        if missing:
            raise ImageRequestError(
                f"{self.mode} requires: " + ", ".join(missing), code="missing_edit_component"
            )
        if self.mode == "inpaint" and self.source is not None and self.mask is not None:
            if self.source.width != self.mask.width or self.source.height != self.mask.height:
                # A mask on different geometry would silently edit the wrong pixels.
                raise ImageRequestError(
                    "inpaint mask dimensions must match the source", code="mask_dimension_mismatch"
                )
        if len(self.instruction) > MAX_INSTRUCTION_CHARS:
            raise ImageRequestError(
                f"instruction must be at most {MAX_INSTRUCTION_CHARS} characters",
                code="invalid_parameter",
            )
        if len(self.prompt) > 4000 or len(self.negative_prompt) > 4000:
            raise ImageRequestError("prompt text must be at most 4000 characters", code="invalid_parameter")
        if (
            isinstance(self.strength, bool)
            or not isinstance(self.strength, (int, float))
            or not 0.0 <= float(self.strength) <= MAX_STRENGTH
        ):
            raise ImageRequestError("strength must be between 0 and 1", code="invalid_parameter")
        if not 64 <= self.width <= 768 or not 64 <= self.height <= 768 or self.width % 8 or self.height % 8:
            raise ImageRequestError(
                "width and height must be multiples of 8 between 64 and 768",
                code="invalid_parameter",
            )
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or not 1 <= self.steps <= 100:
            raise ImageRequestError("steps must be an integer between 1 and 100", code="invalid_parameter")
        if isinstance(self.guidance_scale, bool) or not 0 <= float(self.guidance_scale) <= 30:
            raise ImageRequestError("guidance_scale must be between 0 and 30", code="invalid_parameter")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise ImageRequestError("seed must be an integer or null", code="invalid_parameter")
        if self.model is not None and (
            not isinstance(self.model, str) or not self.model.strip() or len(self.model) > 128
        ):
            raise ImageRequestError(
                "model must be a non-empty string of at most 128 characters", code="invalid_parameter"
            )
        if not isinstance(self.metadata, Mapping):
            raise ImageRequestError("metadata must be an object", code="invalid_parameter")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ImageEditRequest":
        """Build a request from untrusted JSON; references must already be refs."""

        if not isinstance(payload, Mapping):
            raise ImageRequestError("edit request body must be an object")

        def _ref(name: str) -> ImageAssetRef | None:
            value = payload.get(name)
            if value is None:
                return None
            if isinstance(value, ImageAssetRef):
                return value
            if isinstance(value, Mapping):
                try:
                    return ImageAssetRef.from_dict(value)
                except ImageRefError as exc:
                    raise ImageRequestError(
                        f"{name} is not a usable image reference: {exc}", code="invalid_edit_reference"
                    ) from exc
            raise ImageRequestError(f"{name} must be an image reference object", code="invalid_edit_reference")

        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise ImageRequestError("metadata must be an object", code="invalid_parameter")
        return cls(
            mode=str(payload.get("mode", "")),
            prompt=str(payload.get("prompt", "")),
            source=_ref("source"),
            mask=_ref("mask"),
            reference=_ref("reference"),
            instruction=str(payload.get("instruction", "")),
            negative_prompt=str(payload.get("negative_prompt", "")),
            strength=payload.get("strength", 0.6),
            width=payload.get("width", 512),
            height=payload.get("height", 512),
            steps=payload.get("steps", 28),
            guidance_scale=payload.get("guidance_scale", 7.5),
            seed=payload.get("seed"),
            model=payload.get("model"),
            metadata=dict(metadata),
        )

    def as_remote_payload(self) -> dict[str, Any]:
        """Map to the remote editing shape; carries references, never bytes."""

        self.validate()
        payload: dict[str, Any] = {
            "preset_id": self.model,
            "mode": self.mode,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "guidance_scale": float(self.guidance_scale),
            "strength": float(self.strength),
            "init_image_ref": self.source.as_dict() if self.source else None,
            "mask_ref": self.mask.as_dict() if self.mask else None,
            "ip_adapter_ref": self.reference.as_dict() if self.reference else None,
            "instruction": self.instruction or None,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return {key: value for key, value in payload.items() if value is not None}

    def manifest_hint(self) -> dict[str, Any]:
        """Metadata an edited asset should record so the lineage stays traceable."""

        self.validate()
        return {
            "edit_mode": self.mode,
            "source_asset_id": self.source.asset_id if self.source else None,
            "mask_asset_id": self.mask.asset_id if self.mask else None,
            "reference_asset_id": self.reference.asset_id if self.reference else None,
            "strength": float(self.strength),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_schema": EDIT_REQUEST_SCHEMA,
            "mode": self.mode,
            "spec": self.spec.as_dict(),
            "remote_payload": self.as_remote_payload(),
            "manifest_hint": self.manifest_hint(),
        }


@dataclass(frozen=True, slots=True)
class EditAvailability:
    """Whether an editing path can run right now, and why not when it cannot."""

    mode: str
    local_status: str
    remote_status: str
    available: bool
    reasons: tuple[str, ...]
    spec: EditModeSpec

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "local_status": self.local_status,
            "remote_status": self.remote_status,
            "available": self.available,
            "reasons": list(self.reasons),
            "spec": self.spec.as_dict(),
        }


def resolve_edit_availability(
    mode: str,
    *,
    capabilities: ImageAdapterCapabilities | None = None,
) -> EditAvailability:
    """Decide where a mode can run, using the adapter's own advertised facts.

    The local executor is contract-only in this ticket, and a remote path counts as
    available only when an adapter both supports editing and reports a live runtime.
    """

    spec = edit_mode_spec(mode)
    reasons: list[str] = []
    local_status = spec.local_status
    if local_status != "available":
        reasons.append("local_edit_executor_unavailable")

    remote_status = spec.remote_status
    remote_live = bool(
        capabilities is not None
        and capabilities.supports_edit
        and capabilities.runtime_available
    )
    if remote_live:
        remote_status = "available"
    else:
        reasons.append("remote_edit_endpoint_not_verified")
        if capabilities is not None and capabilities.supports_edit and not capabilities.runtime_available:
            reasons.append("adapter_runtime_not_live")

    return EditAvailability(
        mode=mode,
        local_status=local_status,
        remote_status=remote_status,
        available=local_status == "available" or remote_status == "available",
        reasons=tuple(dict.fromkeys(reasons)),
        spec=spec,
    )


def edit_capability_matrix() -> tuple[EditAvailability, ...]:
    """Report every supported editing mode in one stable, comparable order."""

    return tuple(resolve_edit_availability(mode) for mode in EDIT_MODES)
