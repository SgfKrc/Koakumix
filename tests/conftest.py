"""Koakumix 测试公共装置。

- 仓根即 ``harness_workbench`` 包（见 pyproject ``package-dir``）→ 仓根**父目录**入 sys.path；
- 测试间有 ``from .test_x import y`` 的跨模块复用 → 仓根**自身**也入 sys.path；
- 部分用例以子进程运行 ``scripts/*.py`` → 子进程需 PYTHONPATH 指向同一父目录；
- 依赖 QLH 主仓目录结构的契约用例（随迁自 2026-09-15）在独立检出中跳过。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT.parent), str(ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

_existing = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = str(ROOT.parent) + (os.pathsep + _existing if _existing else "")

# 条件跳过：缺资源才跳过（有资源就真跑），不再无条件屏蔽。
# - 双端对照类（rag_baseline / rag_chunking）需要 QLH 主仓（src/rag_store.py）；
# - prompt_cache 需要同仓兄弟子模块 tools/reasonix-codex-bridge 的 prompts。
MAIN_REPO_MODULES = {"test_harness_rag_baseline.py", "test_rag_chunking.py"}
# 仅这两个用例读 bridge 的 prompts；同模块其他用例不依赖，照常运行。
SIBLING_BRIDGE_TESTS = {
    "test_bridge_and_builtin_harness_prompts_are_stable",
    "test_checker_accepts_all_cache_safe_prompt_files",
}


def _main_project_root() -> Path | None:
    for cand in (os.environ.get("QLH_MAIN_PROJECT_ROOT"), str(ROOT.parent)):
        if cand and (Path(cand) / "src" / "rag_store.py").is_file():
            return Path(cand)
    return None


def pytest_configure(config: pytest.Config) -> None:
    root = _main_project_root()
    if root is not None:
        os.environ.setdefault("QLH_MAIN_PROJECT_ROOT", str(root))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    has_main = _main_project_root() is not None
    has_bridge = (ROOT / "tools" / "reasonix-codex-bridge" / "prompts").is_dir()
    for item in items:
        module = Path(str(item.fspath)).name
        if module in MAIN_REPO_MODULES and not has_main:
            item.add_marker(pytest.mark.skip(reason="需 QLH 主仓（QLH_MAIN_PROJECT_ROOT 或同级 src/rag_store.py）"))
        elif item.name in SIBLING_BRIDGE_TESTS and not has_bridge:
            item.add_marker(pytest.mark.skip(reason="需兄弟子模块 tools/reasonix-codex-bridge/prompts"))
