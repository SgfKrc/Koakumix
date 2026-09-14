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

# 这些用例读取主仓（QLH）目录结构中的兄弟资源——reasonix-codex-bridge 的 prompts、
# fixtures/benchmark 的主仓版本、src/rag_store.py 等；在 Koakumix 独立检出中不成立，
# 故跳过（双端对照基线在主仓侧继续维护）。
MAIN_REPO_ONLY_MODULES = {
    "test_harness_benchmark_ledger.py",
    "test_harness_fixture_manifest.py",
    "test_harness_judge_policy.py",
    "test_harness_rag_baseline.py",
    "test_prompt_cache.py",
    "test_rag_chunking.py",
    "test_role_asymmetry.py",
}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip = pytest.mark.skip(
        reason="依赖 QLH 主仓目录结构（随迁用例 2026-09-15）；在 Koakumix 独立检出中跳过"
    )
    for item in items:
        if Path(str(item.fspath)).name in MAIN_REPO_ONLY_MODULES:
            item.add_marker(skip)
