"""``image_generation`` 内置技能——Koakumix 的图像生成能力单元。

复用 ``harness_workbench.image``（契约 + 本地引擎 + 资产库），不重复实现生图后端；
技能只负责**输入契约校验、能力注入检查与结果整形**，使 agent / MCP / CLI 有一致的调用面。
"""
from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Mapping

from ...image.contracts import ImageAdapter, ImageAdapterError, ImageRequest, ImageRequestError
from .. import SkillDefinition, SkillError

SKILL_NAME = "image_generation"

INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "prompt": {"type": "string", "maxLength": 4000, "description": "正向提示词"},
        "negative_prompt": {"type": "string", "maxLength": 4000, "description": "负向提示词"},
        "model": {"type": "string", "maxLength": 128},
        "width": {"type": "integer", "minimum": 64, "maximum": 768},
        "height": {"type": "integer", "minimum": 64, "maximum": 768},
        "steps": {"type": "integer", "minimum": 1, "maximum": 100},
        "guidance_scale": {"type": "number", "minimum": 0, "maximum": 30},
        "seed": {"type": "integer"},
        "response_format": {"type": "string", "enum": ["b64_json", "url"]},
        "user": {"type": "string", "maxLength": 128, "description": "资产归属 scope"},
    },
    "required": ["prompt"],
    "additionalProperties": False,
}

DESCRIPTION = (
    "图像生成（内置技能）：按提示词生成一张图片，返回 b64_json 或资产 URL。"
    "需注入生图适配器（本地引擎或远程服务）；未注入时返回 images_unavailable。"
)


def build_image_generation_skill(*, adapter: ImageAdapter | None = None, store: Any = None) -> SkillDefinition:
    """构造 image_generation 技能（adapter 为生图引擎，store 为资产库）。

    沿用 MCP 端 ``image_generate`` 的结果形状（``{"created", "data": [{"b64_json"|"url"}]}``），
    并额外回报 ``skill`` 与 ``prompt``，便于调用方辨识来源。
    """

    def handler(payload: Mapping[str, Any]) -> dict[str, Any]:
        if adapter is None:
            raise SkillError("images_unavailable", "生图适配器未配置（技能已注册，但未接入本地/远程生图引擎）")
        try:
            request = ImageRequest.from_mapping(payload)
        except ImageRequestError as exc:
            raise SkillError(str(getattr(exc, "code", "invalid_image_request")), str(exc)) from exc
        try:
            generated = adapter.generate(request)
        except ImageAdapterError as exc:
            raise SkillError(str(getattr(exc, "code", "image_backend_error")), str(exc)) from exc

        record = None
        if store is not None:
            record = store.put(generated, prompt=request.prompt, owner_scope=request.user or "local")

        if request.response_format == "url":
            if record is None:
                raise SkillError("image_url_unavailable", "url 响应需要启用资产库（ImageAssetStore）")
            item: dict[str, Any] = {"url": f"/v1/images/assets/{record.asset_id}", "asset_id": record.asset_id}
        else:
            item = {"b64_json": base64.b64encode(generated.data).decode("ascii")}
            if record is not None:
                item["asset_id"] = record.asset_id
        if record is not None:
            item["metadata"] = record.as_dict()

        return {
            "created": int(time.time()),
            "data": [item],
            "skill": SKILL_NAME,
            "prompt": request.prompt,
        }

    return SkillDefinition(
        name=SKILL_NAME,
        title="图像生成",
        description=DESCRIPTION,
        input_schema=INPUT_SCHEMA,
        handler=handler,
        read_only=False,  # 会写入资产库
        docs=str(Path(__file__).with_name("SKILL.md")),
        metadata={"owner": "koakumix", "engine": "harness_workbench.image"},
    )


__all__ = ["INPUT_SCHEMA", "SKILL_NAME", "build_image_generation_skill"]
