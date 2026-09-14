# 技能：image_generation（图像生成）

> 状态：**内置技能（v1）** · 归属：Koakumix · 引擎：`harness_workbench.image`

## 用途

按提示词生成一张图片。这是 Koakumix 的**内置技能形态**——原主项目的 SD 1.5 生图能力
在主仓裁撤（`d18cee7`）后集中到 Koakumix，并由独立"生图工作台"收敛为本技能，
使 agent / MCP 客户端 / CLI 有一致的调用面。

## 输入（JSON Schema 见 `skill.py:INPUT_SCHEMA`）

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `prompt` | string | ≤4000，**必填** | 正向提示词 |
| `negative_prompt` | string | ≤4000 | 负向提示词 |
| `model` | string | ≤128 | 模型标识（适配器解释） |
| `width` / `height` | integer | 64–768 | 尺寸 |
| `steps` | integer | 1–100 | 采样步数（默认 28） |
| `guidance_scale` | number | 0–30 | CFG（默认 7.5） |
| `seed` | integer | — | 随机种子 |
| `response_format` | string | `b64_json` \| `url` | 默认 `b64_json`；`url` 需启用资产库 |
| `user` | string | ≤128 | 资产归属 scope（默认 `local`） |

## 输出

```json
{"created": 1757900000, "skill": "image_generation", "prompt": "……", "data": [{"b64_json": "…", "asset_id": "…", "metadata": {…}}]}
```

`response_format=url` 时 `data[0]` 为 `{"url": "/v1/images/assets/<asset_id>", "asset_id": "…"}`。

## 错误

| code | 场景 |
| --- | --- |
| `images_unavailable` | 未注入生图适配器（技能已注册但引擎未接） |
| `invalid_image_request` / `missing_prompt` | 输入不满足契约 |
| `image_backend_error` | 后端引擎失败 |
| `image_url_unavailable` | 请求 `url` 但未启用资产库 |

## 调用方式

- **MCP**：`skill_list` 列出技能；`skill_run`（`{"name": "image_generation", "arguments": {...}}`）执行
- **CLI**：`koakumix skill list` / `koakumix skill run image_generation --json '{...}'`
- **HTTP**（兼容面）：`POST /v1/images/generations`；能力探测 `GET /v1/images/capabilities`
- **Python**：`from harness_workbench.skills import build_registry; build_registry(image_adapter=...).run("image_generation", {...})`

## 边界

- 技能**不内置模型**：生图引擎由适配器注入（本地 `image.local_engine.LocalImageEngine` 或远程服务），
  因此技能在未配置引擎时是**安全可发现的**（`skill_list` 可见，执行返回 `images_unavailable`）。
- 资产写入由注入的 `ImageAssetStore` 承担；未注入时不落盘。
- QLH 主仓只做可选的多模态**理解**，不保留任何生图面。
