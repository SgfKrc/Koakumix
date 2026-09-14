"""Koakumix 内置技能（skills）测试：注册表 / image_generation 技能 / 错误面。"""
from __future__ import annotations

import base64

import pytest

from harness_workbench.image import (
    GeneratedImage,
    ImageAssetStore,
    LocalImageEngine,
    LocalImageEngineConfig,
)
from harness_workbench.skills import SkillError, SkillRegistry, build_registry
from harness_workbench.skills.image_generation import SKILL_NAME, build_image_generation_skill


class _FakeImageExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object]] = []

    def generate(self, request, manifest):  # noqa: ANN001, ANN201
        self.calls.append((request, manifest))
        return GeneratedImage(b"fake-png", width=request.width, height=request.height, seed=request.seed)


def _adapter(tmp_path):
    (tmp_path / "weights.bin").write_bytes(b"weights")
    import hashlib
    import json

    (tmp_path / ".qlh-sd-asset.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "asset": {"asset_id": "sd15-test", "artifact_id": "sd-test"},
                "files": [
                    {
                        "path": "weights.bin",
                        "size_bytes": 7,
                        "sha256": hashlib.sha256(b"weights").hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return LocalImageEngine(LocalImageEngineConfig(tmp_path, model_id="sd15-test"), executor=_FakeImageExecutor())


def test_registry_contains_image_generation() -> None:
    registry = build_registry()
    assert registry.names() == [SKILL_NAME]
    described = registry.describe()[0]
    assert described["name"] == SKILL_NAME
    assert described["read_only"] is False  # 生图会写资产库
    assert described["docs"].endswith("SKILL.md")


def test_registry_rejects_unknown_skill() -> None:
    with pytest.raises(SkillError) as excinfo:
        build_registry().run("nope", {})
    assert excinfo.value.code == "unknown_skill"


def test_registry_rejects_duplicate_and_bad_name() -> None:
    from harness_workbench.skills import SkillDefinition

    registry = SkillRegistry()
    registry.register(SkillDefinition(name="ok_skill", title="t", description="d", input_schema={}, handler=lambda p: p))
    with pytest.raises(SkillError) as dup:
        registry.register(SkillDefinition(name="ok_skill", title="t", description="d", input_schema={}, handler=lambda p: p))
    assert dup.value.code == "duplicate_skill"
    with pytest.raises(SkillError) as bad:
        SkillDefinition(name="Bad-Name", title="t", description="d", input_schema={}, handler=lambda p: p)
    assert bad.value.code == "invalid_skill_name"


def test_skill_without_adapter_is_discoverable_but_safe() -> None:
    skill = build_image_generation_skill()
    assert skill.name == SKILL_NAME
    with pytest.raises(SkillError) as excinfo:
        skill.handler({"prompt": "a cat"})
    assert excinfo.value.code == "images_unavailable"


def test_skill_generates_b64_and_reports_source(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    registry = build_registry(image_adapter=adapter, image_store=ImageAssetStore(tmp_path / "assets"))
    result = registry.run(SKILL_NAME, {"prompt": "a cat", "width": 64, "height": 64, "seed": 4})
    assert result["skill"] == SKILL_NAME
    assert result["prompt"] == "a cat"
    item = result["data"][0]
    assert base64.b64decode(item["b64_json"]) == b"fake-png"
    assert item["asset_id"]  # 已落资产库


def test_skill_url_requires_store(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    registry = build_registry(image_adapter=adapter)  # 无 store
    with pytest.raises(SkillError) as excinfo:
        registry.run(SKILL_NAME, {"prompt": "a cat", "response_format": "url"})
    assert excinfo.value.code == "image_url_unavailable"
    # 有 store → 返回 url
    with_store = build_registry(image_adapter=adapter, image_store=ImageAssetStore(tmp_path / "assets2"))
    out = with_store.run(SKILL_NAME, {"prompt": "a cat", "response_format": "url"})
    assert out["data"][0]["url"].startswith("/v1/images/assets/")


def test_skill_rejects_invalid_request(tmp_path) -> None:
    registry = build_registry(image_adapter=_adapter(tmp_path))
    with pytest.raises(SkillError) as excinfo:
        registry.run(SKILL_NAME, {"prompt": ""})
    assert excinfo.value.code in {"missing_prompt", "invalid_image_request"}


def test_skill_doc_exists() -> None:
    from pathlib import Path

    skill = build_image_generation_skill()
    doc = Path(skill.docs)
    assert doc.is_file()
    assert "image_generation" in doc.read_text(encoding="utf-8")


def test_mcp_exposes_skill_tools() -> None:
    """MCP 面：skill_list 可枚举，skill_run 未配置引擎时明确报错。"""
    from harness_workbench.mcp_server.builtin import HarnessMCPDependencies, register_builtin_tools
    from harness_workbench.mcp_server.registry import MCPToolError, ToolRegistry

    registry = ToolRegistry()
    definitions = {d.name: d for d in register_builtin_tools(registry, HarnessMCPDependencies())}
    assert "skill_list" in definitions and "skill_run" in definitions
    listed = definitions["skill_list"].handler({})
    assert [s["name"] for s in listed["skills"]] == [SKILL_NAME]
    with pytest.raises(MCPToolError) as excinfo:
        definitions["skill_run"].handler({"name": SKILL_NAME, "arguments": {"prompt": "a cat"}})
    assert excinfo.value.code == "images_unavailable"
    with pytest.raises(MCPToolError) as unknown:
        definitions["skill_run"].handler({"name": "nope", "arguments": {}})
    assert unknown.value.code == "unknown_skill"


def test_cli_skill_list_and_run(capsys) -> None:
    """CLI 面：koakumix skill list 列技能；run 未配置引擎时非零退出并给 code。"""
    from harness_workbench.cli import main

    assert main(["skill", "list"]) == 0
    out = capsys.readouterr().out
    assert SKILL_NAME in out
    assert main(["skill", "run", SKILL_NAME, "--json", '{"prompt": "a cat"}']) == 2
    err = capsys.readouterr().err
    assert "images_unavailable" in err
    assert main(["skill", "run", "nope", "--json", "{}"]) == 2
