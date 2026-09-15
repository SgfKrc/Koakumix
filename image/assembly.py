"""Assemble the local image engine from configuration; fail-closed by default.

KX-IMG-01: the tool surface (MCP ``image_generate``, ``POST /v1/images/generations``)
takes an injected ``image_adapter``.  This module is the single place that decides
what to inject:

* the **real** :class:`~image.diffusers_executor.DiffusersImageExecutor` when the
  API surface explicitly enables it *and* the optional runtime plus a verified
  local SD asset are both present;
* otherwise the documented :class:`~image.local_engine.UnavailableImageExecutor`,
  together with an explicit status and reason -- never a silent fallback that
  pretends the feature works.

Enabling is opt-in (``QLH_HARNESS_IMAGE_EXECUTOR=diffusers``) so a plain checkout
cannot start pulling CUDA stacks into the tool surface by accident.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .contracts import ImageAdapter
from .diffusers_executor import (
    DiffusersExecutorConfig,
    DiffusersImageExecutor,
    diffusers_available,
)
from .local_engine import LocalImageEngine, LocalImageEngineConfig, UnavailableImageExecutor


ASSEMBLY_SCHEMA = "qlh.harness.image_engine_assembly.v1"
ENV_PREFIX = "QLH_HARNESS_IMAGE_"
ENV_ASSET_ROOT = ENV_PREFIX + "ASSET_ROOT"
ENV_DEVICE = ENV_PREFIX + "DEVICE"
ENV_EXECUTOR = ENV_PREFIX + "EXECUTOR"
ENV_MODEL_ID = ENV_PREFIX + "MODEL_ID"

EXECUTOR_DIFFUSERS = "diffusers"
READY = "ready"
ASSET_ROOT_MISSING = "asset_root_not_configured"
EXECUTOR_NOT_ENABLED = "executor_not_enabled"
RUNTIME_MISSING = "diffusers_not_installed"
ASSET_INVALID = "asset_invalid"


@dataclass(frozen=True, slots=True)
class ImageEngineAssembly:
    """The engine to inject plus why it ended up that way."""

    engine: ImageAdapter
    executor_kind: str
    status: str
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == READY and self.executor_kind == EXECUTOR_DIFFUSERS

    def as_dict(self) -> dict[str, Any]:
        return {
            "assembly_schema": ASSEMBLY_SCHEMA,
            "executor_kind": self.executor_kind,
            "status": self.status,
            "ready": self.ready,
            "details": dict(self.details),
        }


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def build_local_image_engine(
    *,
    asset_root: Path | str | None = None,
    device: str | None = None,
    model_id: str | None = None,
    executor: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> ImageEngineAssembly:
    """Build the engine for the tool surface; explicit arguments beat the environment.

    The returned assembly always carries a usable engine object -- when the real
    executor cannot be justified, that object is the documented unavailable one and
    ``status`` says why.
    """

    env = environ if environ is not None else os.environ
    resolved_root = _text(asset_root) or _text(env.get(ENV_ASSET_ROOT))
    resolved_device = _text(device) or _text(env.get(ENV_DEVICE)) or "auto"
    resolved_model = _text(model_id) or _text(env.get(ENV_MODEL_ID)) or None
    resolved_executor = _text(executor) or _text(env.get(ENV_EXECUTOR))

    details: dict[str, Any] = {
        "asset_root_configured": bool(resolved_root),
        "device": resolved_device,
        "model_id": resolved_model,
        "requested_executor": resolved_executor or None,
        "runtime_importable": diffusers_available(),
    }

    def _unavailable(status: str, *, extra: Mapping[str, Any] | None = None) -> ImageEngineAssembly:
        engine = LocalImageEngine(
            LocalImageEngineConfig(
                asset_root=resolved_root or ".",
                model_id=resolved_model,
                backend_id="local_diffusers_txt2img",
            ),
            executor=UnavailableImageExecutor(),
        )
        merged = dict(details)
        merged.update(extra or {})
        return ImageEngineAssembly(
            engine=engine,
            executor_kind="unavailable",
            status=status,
            details=merged,
        )

    if not resolved_root:
        return _unavailable(ASSET_ROOT_MISSING)
    if resolved_executor != EXECUTOR_DIFFUSERS:
        return _unavailable(EXECUTOR_NOT_ENABLED, extra={"hint": f"set {ENV_EXECUTOR}={EXECUTOR_DIFFUSERS} to enable"})
    if not diffusers_available():
        return _unavailable(RUNTIME_MISSING)

    executor_impl = DiffusersImageExecutor(
        DiffusersExecutorConfig(asset_root=resolved_root, device=resolved_device)
    )
    engine = LocalImageEngine(
        LocalImageEngineConfig(
            asset_root=resolved_root,
            model_id=resolved_model,
            backend_id="local_diffusers_txt2img",
        ),
        executor=executor_impl,
    )
    report = engine.inspect()
    merged = dict(details)
    merged["manifest"] = report.as_dict()
    merged["executor"] = executor_impl.status()
    if not report.valid:
        # The executor is real, but the asset is not usable: say so instead of
        # letting the first request discover it.
        merged["manifest_errors"] = list(report.errors)
        return ImageEngineAssembly(
            engine=engine,
            executor_kind=EXECUTOR_DIFFUSERS,
            status=ASSET_INVALID,
            details=merged,
        )
    return ImageEngineAssembly(
        engine=engine,
        executor_kind=EXECUTOR_DIFFUSERS,
        status=READY,
        details=merged,
    )


__all__ = [
    "ASSET_INVALID",
    "ASSET_ROOT_MISSING",
    "ASSEMBLY_SCHEMA",
    "ENV_ASSET_ROOT",
    "ENV_DEVICE",
    "ENV_EXECUTOR",
    "ENV_MODEL_ID",
    "EXECUTOR_DIFFUSERS",
    "EXECUTOR_NOT_ENABLED",
    "ImageEngineAssembly",
    "READY",
    "RUNTIME_MISSING",
    "build_local_image_engine",
]
