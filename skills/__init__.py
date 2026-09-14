"""Koakumix 内置技能（skills）——自包含、可被 agent / MCP / CLI 调用的能力单元。

与 ``tools/`` 的分工：
- ``tools/`` 是开发与实验工具（benchmark、prompt lab、trace replay…）；
- ``skills/`` 是**产品化能力单元**：带 ``SKILL.md`` 说明、JSON Schema 输入契约与稳定执行入口，
  通过 MCP（``skill_list`` / ``skill_run``）与 CLI（``koakumix skill``）暴露给外部使用。

首个内置技能：``image_generation``（图像生成；原独立"生图工作台"收敛为技能形态）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

SkillHandler = Callable[[Mapping[str, Any]], Any]
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


class SkillError(RuntimeError):
    """技能执行失败（可映射为 MCP ``isError`` / CLI 非零退出）。"""

    def __init__(self, code: str, message: str) -> None:
        self.code = str(code)
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class SkillDefinition:
    name: str
    title: str
    description: str
    input_schema: Mapping[str, Any]
    handler: SkillHandler
    read_only: bool = True
    docs: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _NAME_RE.fullmatch(self.name):
            raise SkillError("invalid_skill_name", f"技能名非法: {self.name!r}")
        if not isinstance(self.description, str) or not self.description.strip():
            raise SkillError("invalid_skill_description", f"技能 {self.name} 缺说明")
        if not callable(self.handler):
            raise SkillError("invalid_skill_handler", f"技能 {self.name} handler 不可调用")


class SkillRegistry:
    """技能注册表：注册 / 查询 / 描述 / 执行。"""

    def __init__(self) -> None:
        self._skills: dict[str, SkillDefinition] = {}

    def register(self, skill: SkillDefinition) -> SkillDefinition:
        if skill.name in self._skills:
            raise SkillError("duplicate_skill", f"技能重复注册: {skill.name}")
        self._skills[skill.name] = skill
        return skill

    def get(self, name: str) -> SkillDefinition | None:
        return self._skills.get(str(name))

    def names(self) -> list[str]:
        return sorted(self._skills)

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": s.name,
                "title": s.title,
                "description": s.description,
                "read_only": s.read_only,
                "docs": s.docs,
                "metadata": dict(s.metadata),
            }
            for s in (self._skills[name] for name in self.names())
        ]

    def run(self, name: str, payload: Mapping[str, Any] | None = None) -> Any:
        skill = self.get(name)
        if skill is None:
            raise SkillError("unknown_skill", f"未知技能: {name}（可用: {', '.join(self.names()) or '无'}）")
        return skill.handler(dict(payload or {}))


def build_registry(*, image_adapter: Any = None, image_store: Any = None) -> SkillRegistry:
    """构造内置技能注册表（生图适配器/资产库可选注入）。"""
    from .image_generation import build_image_generation_skill

    registry = SkillRegistry()
    registry.register(build_image_generation_skill(adapter=image_adapter, store=image_store))
    return registry


__all__ = ["SkillDefinition", "SkillError", "SkillRegistry", "build_registry"]
