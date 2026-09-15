"""Real diffusers-backed txt2img executor: lazy, single-loader, CUDA-aware.

KX-IMG-01: the tool surface (MCP ``image_generate`` / ``POST /v1/images/generations``)
already exists and is injection-based, but only :class:`UnavailableImageExecutor`
was wired.  This module provides the *real* executor behind the same
:class:`~image.local_engine.LocalImageExecutor` protocol.

Design boundaries:

* ``torch``/``diffusers`` are imported **lazily**, inside the loader, so the
  harness keeps working (and its tests keep running) without a CUDA stack;
* one pipeline is loaded at most once and every generation is serialized behind a
  single lock -- an 8 GB laptop GPU must not try to hold two pipelines;
* the device is resolved from the actual runtime (``auto`` -> cuda when
  available, else cpu) and recorded on every result, so evidence never claims a
  GPU run that did not happen;
* weights come from the verified local asset directory only.  Nothing is
  downloaded here, and a missing/unverified asset stays a hard error.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Mapping

from .contracts import GeneratedImage, ImageAdapterError, ImageRequest
from .manifest import AssetManifest


DEVICE_CHOICES = ("auto", "cuda", "cpu")
DTYPE_CHOICES = ("float16", "float32")
EXECUTOR_SCHEMA = "qlh.harness.diffusers_executor.v1"


@dataclass(frozen=True, slots=True)
class DiffusersExecutorConfig:
    """How the real executor should behave; every field has a safe default."""

    asset_root: Path | str
    device: str = "auto"
    dtype: str = "float16"
    enable_attention_slicing: bool = True
    pipeline_factory: Callable[..., Any] | None = None

    def __post_init__(self) -> None:
        if self.device not in DEVICE_CHOICES:
            raise ValueError(f"device must be one of: {', '.join(DEVICE_CHOICES)}")
        if self.dtype not in DTYPE_CHOICES:
            raise ValueError(f"dtype must be one of: {', '.join(DTYPE_CHOICES)}")
        if not str(self.asset_root).strip():
            raise ValueError("asset_root is required")


def diffusers_available() -> bool:
    """Report whether the optional runtime is importable, without importing it."""

    import importlib.util

    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("diffusers") is not None
    )


class DiffusersImageExecutor:
    """Load one diffusers pipeline lazily, serialize generation behind a lock.

    Implements the ``LocalImageExecutor`` protocol from ``image.local_engine``:
    ``generate(request, manifest) -> GeneratedImage`` and ``close()``.
    """

    def __init__(self, config: DiffusersExecutorConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._pipeline: Any | None = None
        self._resolved_device: str | None = None
        self._closed = False
        self._load_count = 0
        self._generation_count = 0

    # ---------------------------------------------------------------- status

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def closed(self) -> bool:
        return self._closed

    def status(self) -> dict[str, Any]:
        """Observability for capabilities/evidence; never loads the pipeline."""

        return {
            "executor_schema": EXECUTOR_SCHEMA,
            "kind": "diffusers",
            "device": self._resolved_device or self.config.device,
            "dtype": self.config.dtype,
            "loaded": self.loaded,
            "closed": self._closed,
            "load_count": self._load_count,
            "generation_count": self._generation_count,
            "runtime_importable": diffusers_available(),
        }

    # ---------------------------------------------------------------- device

    def _require_runtime(self) -> tuple[Any, Any]:
        try:
            import torch  # noqa: PLC0415 - deliberately lazy
            from diffusers import StableDiffusionPipeline  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImageAdapterError(
                "diffusers/torch are not installed in this environment",
                code="local_image_runtime_unavailable",
                status_code=503,
            ) from exc
        return torch, StableDiffusionPipeline

    def _resolve_device(self, torch: Any) -> str:
        """Resolve ``auto`` against the real runtime; never assume a GPU."""

        choice = self.config.device
        if choice == "cuda":
            if not torch.cuda.is_available():
                raise ImageAdapterError(
                    "CUDA was requested but is not available in this environment",
                    code="local_image_device_unavailable",
                    status_code=503,
                )
            return "cuda"
        if choice == "cpu":
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------------------------------------------- loading

    def _load_pipeline(self, device: str) -> Any:
        with self._load_lock:
            if self._closed:
                raise ImageAdapterError(
                    "image executor is closed", code="image_executor_closed", status_code=503
                )
            if self._pipeline is not None:
                return self._pipeline
            if self.config.pipeline_factory is not None:
                # Injected factory (tests / alternate runtimes) skips the import.
                self._pipeline = self.config.pipeline_factory(
                    str(self.config.asset_root), device=device, dtype=self.config.dtype
                )
            else:
                torch, pipeline_class = self._require_runtime()
                dtype = torch.float16 if self.config.dtype == "float16" else torch.float32
                if device == "cpu" and dtype is torch.float16:
                    # fp16 on CPU is either unsupported or pathologically slow.
                    dtype = torch.float32
                try:
                    self._pipeline = pipeline_class.from_pretrained(
                        str(self.config.asset_root),
                        torch_dtype=dtype,
                        local_files_only=True,
                    )
                except Exception as exc:
                    raise ImageAdapterError(
                        "local SD asset could not be loaded by diffusers",
                        code="local_image_load_failed",
                        status_code=502,
                    ) from exc
            if self.config.enable_attention_slicing:
                enable = getattr(self._pipeline, "enable_attention_slicing", None)
                if callable(enable):
                    try:
                        enable()
                    except Exception:  # pragma: no cover - pipeline dependent
                        pass
            move = getattr(self._pipeline, "to", None)
            if callable(move):
                move(device)
            self._resolved_device = device
            self._load_count += 1
            return self._pipeline

    # -------------------------------------------------------------- generate

    def generate(self, request: ImageRequest, manifest: AssetManifest) -> GeneratedImage:
        """Run one txt2img pass against the verified local asset."""

        if self._closed:
            raise ImageAdapterError(
                "image executor is closed", code="image_executor_closed", status_code=503
            )
        device = self.config.device
        torch: Any | None = None
        if self.config.pipeline_factory is None:
            torch, _ = self._require_runtime()
            device = self._resolve_device(torch)
        elif device == "auto":
            device = "cpu"

        # One pipeline, one generation at a time: two concurrent passes on an
        # 8 GB card would OOM before either finishes.
        with self._lock:
            pipeline = self._load_pipeline(device)
            try:
                result = self._invoke(pipeline, request, torch, device)
            except ImageAdapterError:
                raise
            except Exception as exc:
                raise ImageAdapterError(
                    "diffusers pipeline failed during generation",
                    code="local_image_generation_failed",
                    status_code=502,
                ) from exc
            self._generation_count += 1
            return self._encode(result, request, manifest, device)

    def _invoke(self, pipeline: Any, request: ImageRequest, torch: Any | None, device: str) -> Any:
        kwargs: dict[str, Any] = {
            "prompt": request.prompt,
            "width": request.width,
            "height": request.height,
            "num_inference_steps": request.steps,
            "guidance_scale": float(request.guidance_scale),
        }
        if request.negative_prompt:
            kwargs["negative_prompt"] = request.negative_prompt
        if request.seed is not None and torch is not None:
            kwargs["generator"] = torch.Generator(device=device).manual_seed(int(request.seed))
        return pipeline(**kwargs)

    @staticmethod
    def _encode(result: Any, request: ImageRequest, manifest: AssetManifest, device: str) -> GeneratedImage:
        images = getattr(result, "images", None)
        if not isinstance(images, (list, tuple)) or not images:
            raise ImageAdapterError(
                "diffusers pipeline returned no image",
                code="invalid_image_result",
                status_code=502,
            )
        first = images[0]
        buffer = BytesIO()
        try:
            first.save(buffer, format="PNG")
        except Exception as exc:  # pragma: no cover - PIL dependent
            raise ImageAdapterError(
                "generated image could not be encoded as PNG",
                code="invalid_image_result",
                status_code=502,
            ) from exc
        data = buffer.getvalue()
        width = getattr(first, "width", request.width)
        height = getattr(first, "height", request.height)
        return GeneratedImage(
            data=data,
            mime_type="image/png",
            width=width if isinstance(width, int) else request.width,
            height=height if isinstance(height, int) else request.height,
            seed=request.seed,
            metadata={
                "executor_schema": EXECUTOR_SCHEMA,
                "device": device,
                "asset_id": manifest.asset_id,
                "artifact_id": manifest.artifact_id,
                "steps": request.steps,
                "guidance_scale": float(request.guidance_scale),
            },
        )

    # ----------------------------------------------------------------- close

    def close(self) -> None:
        """Release the pipeline exactly once; safe to call repeatedly."""

        with self._load_lock:
            if self._closed:
                return
            self._closed = True
            pipeline = self._pipeline
            self._pipeline = None
            if pipeline is not None:
                try:
                    move = getattr(pipeline, "to", None)
                    if callable(move):  # pragma: no cover - pipeline dependent
                        move("cpu")
                except Exception:
                    pass
            if diffusers_available():
                try:  # pragma: no cover - environment dependent
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

    def __enter__(self) -> "DiffusersImageExecutor":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def executor_evidence(executor: DiffusersImageExecutor | None) -> Mapping[str, Any]:
    """Evidence block for capability reporting; ``None`` means "not wired"."""

    if executor is None:
        return {"executor": None, "runtime_importable": diffusers_available()}
    return {"executor": executor.status(), "runtime_importable": diffusers_available()}


__all__ = [
    "DEVICE_CHOICES",
    "DTYPE_CHOICES",
    "DiffusersExecutorConfig",
    "DiffusersImageExecutor",
    "diffusers_available",
    "executor_evidence",
]
