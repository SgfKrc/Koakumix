<p align="center">
  <img src="assets/Koakumix.png" alt="Koakumix" width="220">
</p>

# Koakumix

> **Language**: 简体中文 · [English](README.en.md)

Koakumix 是 QLH 项目的**小模型本地增强对话 harness**（原 `harness_workbench/`，2026-09-15 迁出为独立子模块）。

价值主张：**同一模型，在 Koakumix 里跑应明显强于普通对话**——通过长上下文管理、跨会话长期记忆、检索增强（RAG）与小模型组合（draft-verify）把本地小模型的可用性拉高一档。不对标通用 harness，只做"本地小模型增强"这一件事。

## 能力

| 模块 | 职责 |
| --- | --- |
| `api_layer/` | OpenAI 兼容 HTTP 编排（chat + RAG + memory + 压缩引擎） |
| `context_engine/` | 上下文预算与策略（长上下文管理） |
| `memory/` | 跨会话长期记忆（提取/存储/检索） |
| `rag/` | 分块策略、索引粒度与离线基线 |
| `eval/` `research/` | 评测夹具、红队契约、角色非对称报告 |
| `image/` | **图像生成**（本地引擎 + 资产 manifest/契约） |
| `mcp_server/` | MCP 工具面（含 `image_generate`） |
| `model_profiles/` | 模型画像与小模型档位 |
| `cli.py` `tui.py` | 命令行与 TUI 入口 |
| `desktop.py` | **桌面壳**（pywebview：本地起 API 并把系统 WebView 指向它；`desktop` extra 可选，窗口图标取 `assets/Koakumix.png`） |
| `ui_react/` | 可选 Web UI（`node_modules` 本地重建，不入库） |

**图像生成的唯一归属在本仓库**（QLH 主仓已于 `d18cee7` 裁撤 SD 1.5 全链，只保留可选的多模态**理解**）。

## 快速开始

```bash
pip install -e ".[tui]"          # 可编辑安装（含 Textual TUI）
koakumix --help                  # CLI
python -m harness_workbench.tui  # TUI
```

MCP 工具面（stdio）：

```bash
python -m harness_workbench.mcp_server
```

桌面壳（pywebview）：

```bash
cd ui_react && npm install && npm run build    # 先构建前端（dist 不入库）
pip install -e ".[desktop]"                     # 可选依赖：pywebview + fastapi + uvicorn
```

手动启动（三种任选）：

```bash
# 1) 双击本文件，或从任意目录调用；它自行切到仓库根并选用 .venv-test：
koakumix-desktop.cmd

# 2) 从仓库根以模块方式启动（最稳，不依赖安装状态）：
python -m harness_workbench.desktop --model models/qwen3-0.6b-q8_0.gguf

# 3) editable 安装成功后的入口脚本（任意目录可用）：
koakumix-desktop --check                        # 就绪自检（不开窗、不需要模型）
koakumix-desktop --model path/to/model.gguf     # 起本地 API 并打开桌面窗口
```

后端形态（默认自动选择）：

- `--backend qlh`（默认）：先探测 `--qlh-base-url`（默认 `http://127.0.0.1:8090`）上的 QLH 主项目 API；通则直连，模型选择 / 加载 / 设备画像面板因此有数据。
- 探测失败则**自动回退**自起 llama-server（需 `--model`），工作台仍能打开，但只有 chat 与对话相关能力。
- `--backend llama` 可强制走回退路径。

数据（会话 / 记忆 / RAG / 图片）默认落在 `%LOCALAPPDATA%\Koakumix`，可用 `--data-dir` 覆盖。

## 测试

```bash
python -m pytest tests -q
```

## 与 QLH 主仓的关系

Koakumix 作为 **git submodule** 挂在 QLH 主仓的 `harness_workbench/` 路径：

- 代码层**零耦合**：主仓不 `import` 本仓库任何模块（仅子模块指针 + 一条生图面守卫校验）；
- 接入方式：主仓通过 **OpenAI 兼容 API**、模型 manifest 与显式工件引用消费本仓库能力；
- 共享资产（prompt sets、rubrics、RAG 基准夹具）以**契约**形式同步，不作隐式 import。

## 迁移沿革

- 2026-09-14：QLH 主仓内建立 harness（`harness_workbench/`），定位为小模型增强 harness；
- 2026-09-15：以 `git subtree split` 迁出为独立仓库（**保留 864 提交历史**），主仓转为 submodule；随迁 39 个测试、4 个门面脚本（`qlh_say` / `model_quotes` / `check_prompt_cache` / `role_asymmetry_report`）与 `fixtures/benchmark` 副本；
- QLH 主仓生图裁撤（`d18cee7`）后，**图像生成能力集中到本仓库**。

## 许可证

MIT License（见 [LICENSE](LICENSE)）。
