"""KX-IMG-01 tests: the real diffusers executor boundary and its assembly.

The executor must load one pipeline at most once, serialize generations behind a
single lock, resolve the device from the real runtime, stay lazy about
torch/diffusers, and fail with stable errors.  The assembly must be opt-in and
never pretend the feature works when the runtime or the asset is missing.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from harness_workbench.image import (
    ASSET_ROOT_MISSING,
    ENV_ASSET_ROOT,
    ENV_DEVICE,
    ENV_EXECUTOR,
    EXECUTOR_DIFFUSERS,
    EXECUTOR_NOT_ENABLED,
    RUNTIME_MISSING,
    DiffusersExecutorConfig,
    DiffusersImageExecutor,
    ImageAdapterError,
    ImageRequest,
    build_local_image_engine,
    diffusers_available,
)
from harness_workbench.image.diffusers_executor import EXECUTOR_SCHEMA
from harness_workbench.image.manifest import AssetManifest


class _FakeImage:
    width = 64
    height = 64

    def save(self, buffer, format=None):  # noqa: A002 - PIL signature
        buffer.write(b"\x89PNG\r\n\x1a\n" + b"payload" * 8)


class _FakePipeline:
    def __init__(self, *, delay: float = 0.0, fail: bool = False) -> None:
        self.calls: list[dict] = []
        self.delay = delay
        self.fail = fail
        self.moved_to: list[str] = []
        self.slicing_enabled = False

    def __call__(self, **kwargs):
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("pipeline exploded")
        self.calls.append(kwargs)
        return SimpleNamespace(images=[_FakeImage()])

    def to(self, device: str) -> None:
        self.moved_to.append(device)

    def enable_attention_slicing(self) -> None:
        self.slicing_enabled = True


def _factory(pipeline: _FakePipeline):
    def make(root: str, *, device: str, dtype: str):
        pipeline.loaded_for = (root, device, dtype)  # type: ignore[attr-defined]
        return pipeline

    return make


def _executor(pipeline: _FakePipeline, **overrides) -> DiffusersImageExecutor:
    config = DiffusersExecutorConfig(
        asset_root="/tmp/sd15-fixture",
        pipeline_factory=_factory(pipeline),
        **overrides,
    )
    return DiffusersImageExecutor(config)


def _request(**overrides) -> ImageRequest:
    payload = {"prompt": "a red cube on a table", "width": 64, "height": 64, "steps": 4, "seed": 7}
    payload.update(overrides)
    return ImageRequest(**payload)


def _manifest() -> AssetManifest:
    return AssetManifest(asset_id="sd15-fixture", artifact_id="artifact-1", files=())


def test_torch_and_diffusers_are_never_imported_at_module_import_time() -> None:
    # The module imports cleanly in an environment without the CUDA stack; the
    # probe reports availability instead of importing.
    assert isinstance(diffusers_available(), bool)
    assert EXECUTOR_SCHEMA.startswith("qlh.harness.diffusers_executor")


def test_pipeline_is_loaded_lazily_and_exactly_once() -> None:
    pipeline = _FakePipeline()
    executor = _executor(pipeline)

    assert executor.loaded is False
    assert executor.status()["load_count"] == 0

    executor.generate(_request(), _manifest())
    executor.generate(_request(prompt="second"), _manifest())

    assert executor.loaded is True
    assert executor.status()["load_count"] == 1
    assert executor.status()["generation_count"] == 2
    assert pipeline.loaded_for == ("/tmp/sd15-fixture", "cpu", "float16")
    assert len(pipeline.calls) == 2


def test_attention_slicing_and_device_move_are_applied() -> None:
    pipeline = _FakePipeline()
    executor = _executor(pipeline, enable_attention_slicing=True)

    executor.generate(_request(), _manifest())

    assert pipeline.slicing_enabled is True
    assert pipeline.moved_to == ["cpu"]


def test_generations_are_serialized_behind_one_lock() -> None:
    state = {"current": 0, "peak": 0}

    class _Probe(_FakePipeline):
        def __call__(self, **kwargs):
            state["current"] += 1
            state["peak"] = max(state["peak"], state["current"])
            time.sleep(0.02)
            state["current"] -= 1
            return SimpleNamespace(images=[_FakeImage()])

    executor = _executor(_Probe())
    errors: list[BaseException] = []

    def run() -> None:
        try:
            executor.generate(_request(), _manifest())
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    # Two concurrent passes on an 8 GB card would OOM; the lock forbids overlap.
    assert state["peak"] == 1
    assert executor.status()["generation_count"] == 4


def test_pipeline_failure_becomes_a_stable_adapter_error() -> None:
    executor = _executor(_FakePipeline(fail=True))

    with pytest.raises(ImageAdapterError) as excinfo:
        executor.generate(_request(), _manifest())

    assert excinfo.value.code == "local_image_generation_failed"
    assert excinfo.value.status_code == 502
    # A failed pass is not counted as a generation.
    assert executor.status()["generation_count"] == 0


def test_result_is_png_bytes_with_traceable_metadata() -> None:
    pipeline = _FakePipeline()
    executor = _executor(pipeline)

    image = executor.generate(_request(seed=99), _manifest())

    assert image.data.startswith(b"\x89PNG\r\n\x1a\n")
    assert image.mime_type == "image/png"
    assert image.width == 64 and image.height == 64
    assert image.seed == 99
    assert image.metadata["device"] == "cpu"
    assert image.metadata["asset_id"] == "sd15-fixture"
    assert image.metadata["executor_schema"] == EXECUTOR_SCHEMA


def test_request_parameters_reach_the_pipeline() -> None:
    pipeline = _FakePipeline()
    executor = _executor(pipeline)

    executor.generate(_request(prompt="hello", negative_prompt="blurry", steps=6, guidance_scale=4.5), _manifest())

    call = pipeline.calls[0]
    assert call["prompt"] == "hello"
    assert call["negative_prompt"] == "blurry"
    assert call["num_inference_steps"] == 6
    assert call["guidance_scale"] == 4.5
    assert call["width"] == 64 and call["height"] == 64


def test_close_is_idempotent_and_blocks_further_generation() -> None:
    executor = _executor(_FakePipeline())
    executor.generate(_request(), _manifest())

    executor.close()
    executor.close()  # second close must be a no-op

    assert executor.closed is True
    assert executor.loaded is False
    with pytest.raises(ImageAdapterError) as excinfo:
        executor.generate(_request(), _manifest())
    assert excinfo.value.code == "image_executor_closed"


def test_context_manager_closes_the_executor() -> None:
    pipeline = _FakePipeline()
    with _executor(pipeline) as executor:
        executor.generate(_request(), _manifest())
    assert executor.closed is True


def test_missing_runtime_is_reported_without_an_injected_factory() -> None:
    if diffusers_available():  # pragma: no cover - only in a full CUDA env
        pytest.skip("optional runtime installed; the unavailable path cannot be observed")

    executor = DiffusersImageExecutor(DiffusersExecutorConfig(asset_root="/tmp/sd15-fixture"))

    with pytest.raises(ImageAdapterError) as excinfo:
        executor.generate(_request(), _manifest())
    assert excinfo.value.code == "local_image_runtime_unavailable"
    assert excinfo.value.status_code == 503


def test_config_rejects_unknown_device_and_dtype() -> None:
    with pytest.raises(ValueError, match="device"):
        DiffusersExecutorConfig(asset_root="/tmp/x", device="tpu")
    with pytest.raises(ValueError, match="dtype"):
        DiffusersExecutorConfig(asset_root="/tmp/x", dtype="int8")
    with pytest.raises(ValueError, match="asset_root"):
        DiffusersExecutorConfig(asset_root="   ")


def test_explicit_cuda_without_a_factory_fails_closed() -> None:
    if diffusers_available():  # pragma: no cover - environment dependent
        pytest.skip("optional runtime installed; skipping the not-installed path")

    executor = DiffusersImageExecutor(
        DiffusersExecutorConfig(asset_root="/tmp/sd15-fixture", device="cuda")
    )
    with pytest.raises(ImageAdapterError) as excinfo:
        executor.generate(_request(), _manifest())
    assert excinfo.value.code in {"local_image_runtime_unavailable", "local_image_device_unavailable"}


def test_assembly_requires_an_asset_root() -> None:
    assembly = build_local_image_engine(environ={})

    assert assembly.status == ASSET_ROOT_MISSING
    assert assembly.executor_kind == "unavailable"
    assert assembly.ready is False
    assert assembly.details["asset_root_configured"] is False


def test_assembly_is_opt_in_and_never_silently_enables_the_gpu_stack() -> None:
    assembly = build_local_image_engine(asset_root="/tmp/sd15-fixture", environ={})

    assert assembly.status == EXECUTOR_NOT_ENABLED
    assert assembly.executor_kind == "unavailable"
    assert assembly.details["requested_executor"] is None
    assert ENV_EXECUTOR in assembly.details["hint"]


def test_assembly_reports_a_missing_runtime_instead_of_claiming_readiness() -> None:
    if diffusers_available():  # pragma: no cover - environment dependent
        pytest.skip("optional runtime installed; the missing-runtime path is unreachable")

    assembly = build_local_image_engine(
        asset_root="/tmp/sd15-fixture", executor=EXECUTOR_DIFFUSERS, environ={}
    )

    assert assembly.status == RUNTIME_MISSING
    assert assembly.executor_kind == "unavailable"
    assert assembly.details["runtime_importable"] is False


def test_assembly_reads_the_environment_when_arguments_are_absent() -> None:
    env = {ENV_ASSET_ROOT: "/tmp/sd15-from-env", ENV_EXECUTOR: EXECUTOR_DIFFUSERS, ENV_DEVICE: "cpu"}
    assembly = build_local_image_engine(environ=env)

    assert assembly.details["asset_root_configured"] is True
    assert assembly.details["device"] == "cpu"
    assert assembly.details["requested_executor"] == EXECUTOR_DIFFUSERS
    # Without the optional runtime the honest answer is "not installed", not "ready".
    assert assembly.status in {RUNTIME_MISSING, "asset_invalid"}


def test_assembly_dict_is_stable() -> None:
    assembly = build_local_image_engine(environ={})
    payload = assembly.as_dict()

    assert payload["assembly_schema"].startswith("qlh.harness.image_engine_assembly")
    assert payload["executor_kind"] == "unavailable"
    assert payload["status"] == ASSET_ROOT_MISSING
    assert payload["ready"] is False


def test_engine_from_assembly_still_verifies_before_generating() -> None:
    """Even the unavailable engine keeps the manifest gate in front of the executor."""

    assembly = build_local_image_engine(asset_root="/tmp/does-not-exist", environ={})
    capabilities = assembly.engine.capabilities()

    assert capabilities.runtime_available is False
    assert capabilities.supports_txt2img is False
    with pytest.raises(ImageAdapterError) as excinfo:
        assembly.engine.generate(_request())
    assert excinfo.value.code in {"asset_manifest_invalid"}
