"""SIDE-KOAKU-02: 图像生成归 Koakumix —— 本仓库侧的回归与归属守卫。

主仓侧的裁撤守卫（`test_image_generation_cull.py`）随产品壳迁到了 `qlh-shell`；本文件
承担**本仓库这一侧**的责任：证明生图能力**在这里**、端到端可用，并在将来被误删/误迁时
立刻失败。

覆盖：
1. 真执行器（`DiffusersImageExecutor` + 注入的 fake pipeline）经 `/v1/images/generations` 端到端；
2. 未装配真实执行器时 fail-closed（不谎称能生图）；
3. 同一 adapter 经 MCP `image_generate` 工具可调用；
4. 执行器失败时返回稳定错误，不泄漏库异常；
5. 归属守卫：`image/` 模块与 API 端点必须存在，且本仓库不 import 主仓 `src/`。
"""

from __future__ import annotations

import base64
import hashlib
import json
import pathlib

import pytest

from harness_workbench.adapters.base import (
    AdapterCapabilities,
    AdapterModel,
    AdapterRequest,
    AdapterResponse,
    StreamChunk,
)
from harness_workbench.api_layer import create_app
from harness_workbench.image import (
    DiffusersExecutorConfig,
    DiffusersImageExecutor,
    ImageAdapterError,
    ImageAssetStore,
    ImageRequest,
    LocalImageEngine,
    LocalImageEngineConfig,
)
from harness_workbench.mcp_server import HarnessMCPDependencies, create_harness_server


# --------------------------------------------------------------------- fixtures
class _FakePIL:
    width = 512
    height = 512

    def save(self, buffer, format=None):  # noqa: A002 - PIL signature
        buffer.write(b"\x89PNG\r\n\x1a\n" + b"koakumix" * 4)


class _FakePipeline:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[dict] = []
        self.fail = fail

    def __call__(self, **kwargs):
        if self.fail:
            raise RuntimeError("diffusers exploded")
        self.calls.append(kwargs)
        return type("R", (), {"images": [_FakePIL()]})()

    def to(self, device):
        return None

    def enable_attention_slicing(self):
        return None


def _pipeline_factory(pipeline: _FakePipeline):
    def make(root, *, device, dtype, variant=None):
        return pipeline

    return make


def _asset_root(tmp_path: pathlib.Path, *, asset_id: str = "sd15-koakumix") -> pathlib.Path:
    root = tmp_path / "sd15-koakumix"
    root.mkdir(parents=True, exist_ok=True)
    content = b"weights"
    (root / "weights.bin").write_bytes(content)
    (root / ".qlh-sd-asset.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "asset": {"asset_id": asset_id, "artifact_id": "koakumix-test"},
                "files": [
                    {
                        "path": "weights.bin",
                        "size_bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


def _real_engine(tmp_path: pathlib.Path, pipeline: _FakePipeline) -> LocalImageEngine:
    """A LocalImageEngine running the *real* executor, fed an offline fake pipeline."""

    root = _asset_root(tmp_path)
    executor = DiffusersImageExecutor(
        DiffusersExecutorConfig(asset_root=str(root), pipeline_factory=_pipeline_factory(pipeline))
    )
    return LocalImageEngine(LocalImageEngineConfig(root, model_id="sd15-koakumix"), executor=executor)


class _ChatAdapter:
    def capabilities(self):
        return AdapterCapabilities(backend="fake")

    def models(self):
        return (AdapterModel("fake"),)

    def complete(self, request: AdapterRequest):
        return AdapterResponse("id", request.model, "ok")

    def stream(self, request: AdapterRequest):
        yield StreamChunk("id", request.model, {"content": "ok"}, "stop")

    def close(self):
        return None


# ------------------------------------------------- 1) API 端到端（真执行器路径）
def test_api_generation_goes_through_the_real_executor(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    pipeline = _FakePipeline()
    engine = _real_engine(tmp_path, pipeline)
    store = ImageAssetStore(tmp_path / "assets")
    client = TestClient(create_app(_ChatAdapter(), image_adapter=engine, image_store=store))

    response = client.post(
        "/v1/images/generations",
        json={"prompt": "a koakumix mascot", "size": "512x512", "steps": 4, "seed": 11},
    )

    assert response.status_code == 200
    payload = response.json()["data"][0]
    decoded = base64.b64decode(payload["b64_json"])
    assert decoded.startswith(b"\x89PNG\r\n\x1a\n")
    # The image really travelled through the executor, not a stub.
    assert len(pipeline.calls) == 1
    assert pipeline.calls[0]["prompt"] == "a koakumix mascot"
    assert pipeline.calls[0]["num_inference_steps"] == 4
    # ...and was stored, so the asset route can serve it back.
    asset_id = payload["asset_id"]
    assert client.get(f"/v1/images/assets/{asset_id}").content == decoded


def test_capabilities_endpoint_reflects_the_real_engine(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    engine = _real_engine(tmp_path, _FakePipeline())
    client = TestClient(create_app(_ChatAdapter(), image_adapter=engine))

    caps = client.get("/v1/images/capabilities").json()
    assert caps["runtime_available"] is True
    assert caps["supports_txt2img"] is True


# ------------------------------------------------------------ 2) fail-closed 面
def test_unconfigured_engine_never_claims_it_can_generate(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    root = _asset_root(tmp_path)
    engine = LocalImageEngine(LocalImageEngineConfig(root, model_id="sd15-koakumix"))  # no executor
    client = TestClient(create_app(_ChatAdapter(), image_adapter=engine))

    assert client.get("/v1/images/capabilities").json()["supports_txt2img"] is False
    failed = client.post("/v1/images/generations", json={"prompt": "nope"})
    assert failed.status_code >= 400


def test_api_without_an_image_adapter_is_explicitly_unavailable():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    client = TestClient(create_app(_ChatAdapter()))
    response = client.post("/v1/images/generations", json={"prompt": "nope"})
    assert response.status_code >= 400
    assert "images_unavailable" in response.text or "unavailable" in response.text.lower()


# ------------------------------------------------------------------ 3) MCP 工具
def test_mcp_image_generate_uses_the_same_adapter(tmp_path):
    pipeline = _FakePipeline()
    engine = _real_engine(tmp_path, pipeline)
    store = ImageAssetStore(tmp_path / "assets")
    server = create_harness_server(
        dependencies=HarnessMCPDependencies(image_adapter=engine, image_store=store)
    )

    listed = {tool["name"] for tool in server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]}
    assert {"image_generate", "image_capabilities"} <= listed

    call = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "image_generate", "arguments": {"prompt": "via mcp", "width": 512, "height": 512}},
        }
    )
    assert call["result"].get("isError") is not True
    assert len(pipeline.calls) == 1
    assert pipeline.calls[0]["prompt"] == "via mcp"


# ------------------------------------------------------------- 4) 稳定错误映射
def test_executor_failure_surfaces_as_a_stable_error(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    engine = _real_engine(tmp_path, _FakePipeline(fail=True))
    client = TestClient(create_app(_ChatAdapter(), image_adapter=engine), raise_server_exceptions=False)

    response = client.post("/v1/images/generations", json={"prompt": "will fail"})
    assert response.status_code >= 400
    body = response.text.lower()
    assert "diffusers exploded" not in body  # the library exception must not leak
    assert "traceback" not in body


def test_engine_rejects_a_model_it_does_not_serve(tmp_path):
    engine = _real_engine(tmp_path, _FakePipeline())
    request = ImageRequest.from_mapping({"prompt": "x", "model": "some-other-model"})

    with pytest.raises(ImageAdapterError) as excinfo:
        engine.generate(request)
    assert excinfo.value.code == "model_not_available"


# ---------------------------------------------------------------- 5) 归属守卫
def test_image_generation_lives_in_this_repository():
    """若生图模块或端点被误删/误迁，本仓库的回归必须立刻失败。"""

    repo = pathlib.Path(__file__).resolve().parents[1]
    for relative in (
        "image/__init__.py",
        "image/assets.py",
        "image/contracts.py",
        "image/diffusers_executor.py",
        "image/assembly.py",
        "image/editing.py",
        "image/refs.py",
    ):
        assert (repo / relative).is_file(), f"missing {relative}"

    api_source = (repo / "api_layer" / "app.py").read_text(encoding="utf-8")
    assert "/v1/images/generations" in api_source
    mcp_source = (repo / "mcp_server" / "builtin.py").read_text(encoding="utf-8")
    assert "image_generate" in mcp_source


def test_harness_runtime_does_not_import_the_main_repository():
    """零耦合约束：**运行时模块**不得 import 主仓 `src/`。

    用 AST 而非字符串匹配，避免注释/文档字样造成误报。边界如实登记：
    - `tools/rag_baseline.py` 是**唯一**的延迟导入（双端基准对照），已登记为例外；
    - `tests/` 下的条件依赖（例如 `test_rag_chunking.py` 在主仓可见时对照主仓实现）
      属于测试面、独立检出会跳过，不在本守卫的范围内。
    """

    import ast

    repo = pathlib.Path(__file__).resolve().parents[1]
    allowed = {"tools/rag_baseline.py"}
    offenders: list[str] = []

    for path in sorted(repo.rglob("*.py")):
        relative = path.relative_to(repo).as_posix()
        if relative.startswith("tests/") or relative in allowed:
            continue
        if any(part in {"__pycache__", "node_modules", ".venv"} for part in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - 正常仓库不会出现
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(name == "src" or name.startswith("src.") for name in names):
                offenders.append(f"{relative}:{node.lineno}")

    assert offenders == []
