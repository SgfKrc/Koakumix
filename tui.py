"""Koakumix CLI 工作台 —— Textual TUI。

配色分两层（用户指定）：
- **启动页**保留恐虐红金白黑（见 :mod:`harness_workbench.splash`）；
- **主界面**用中性深色（参照 reasonix cli 的 GitHub Dark 观感：底 #0d1117 / #161b22、
  正文 #c9d1d9、次要 #8b949e、强调紫 #bc8cff 与蓝 #58a6ff）—— 大面积红底黑字长时间
  阅读很吃力。

后端连接：默认**先复用**已有 harness API；若 ``--host`` 无人应答且未禁用，则**自动拉起**
一个内嵌 harness API（复用 :mod:`harness_workbench.desktop` 的装配：QLH 主项目优先，
否则自起 llama-server）。避免一进来就退化成 FIXTURE 离线态。

左栏导航切换主区内容：
- 对话页：新消息自动滚动到底（否则长会话只看到顶部那截，像"没有回应"）；
- 知识库页：**可检索**（``POST /v1/rag/search``）**可入库**（``POST /v1/rag/sources``）；
- MCP 页：列出内置工具（``GET /v1/mcp/tools``），选中后按 JSON 参数调用（``/v1/mcp/call``）；
- 资产页：给提示词**生成图片**（``POST /v1/images/generations``）。终端无法内嵌显示图片，
  因此生成结果**落盘**并在面板上报告路径与元数据；
- 模型库页：画像 / 预设 / 下载队列（``/v1/model-profiles``、``/v1/model-presets``、
  ``/v1/model-downloads``），选中预设按 d 入队下载。

启动动画与 ``--no-splash`` / ``--splash-time`` 语义不变；非 TTY 自动跳过。
"""

from __future__ import annotations

import argparse
import base64
import json
import threading
import time as _time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_HOST = "http://127.0.0.1:8090"
QLH_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_IMAGE_SIZE = "512x512"
DEFAULT_IMAGE_STEPS = 28
MIME_SUFFIXES = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}

NAV_ITEMS: tuple[tuple[str, str], ...] = (
    ("chat", "对话"),
    ("library", "知识库"),
    ("mcp", "MCP"),
    ("assets", "资产"),
    ("presets", "模型库"),
    ("runtime", "运行时"),
)

RAG_INDEX_MODES: tuple[str, ...] = ("fts", "keyword", "graph")
HIT_SNIPPET_CHARS = 240
# 入库单个文件的上限：防止误指到一个巨大文件把库撑爆。
MAX_SOURCE_BYTES = 4 * 1024 * 1024


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Koakumix CLI 工作台")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"harness API base URL（默认 {DEFAULT_HOST}）")
    parser.add_argument("--model", default=None, help="自起后端时使用的 GGUF 模型路径")
    parser.add_argument("--llama-executable", default=None, help="自起后端时的 llama-server 可执行文件")
    parser.add_argument("--no-serve", action="store_true", help="不自动拉起后端（只连 --host）")
    parser.add_argument("--no-splash", action="store_true", help="跳过启动动画")
    parser.add_argument("--splash-time", type=float, default=1.0, help="启动动画最小展示秒数（默认 1.0）")
    parser.add_argument("--rag-limit", type=int, default=8, help="知识库检索返回条数上限（默认 8）")
    parser.add_argument("--rag-mode", default="fts", choices=RAG_INDEX_MODES, help="知识库检索模式（默认 fts）")
    parser.add_argument("--image-dir", default=None, help="生成图片的落盘目录（默认 %%LOCALAPPDATA%%\\Koakumix\\images）")
    parser.add_argument("--image-size", default=DEFAULT_IMAGE_SIZE, help=f"生成尺寸 WxH（默认 {DEFAULT_IMAGE_SIZE}）")
    parser.add_argument("--image-steps", type=int, default=DEFAULT_IMAGE_STEPS, help=f"采样步数（默认 {DEFAULT_IMAGE_STEPS}）")
    return parser


def _request_json(
    host: str, path: str, payload: dict[str, Any] | None = None, *, timeout: float = 8.0
) -> dict[str, Any]:
    request = urllib.request.Request(
        host.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # HTTPError is a URLError subclass, so it must be caught first.  The API reports
        # failures as {"error": {...}} -- surface that instead of a bare status code.
        detail = ""
        try:
            body = json.loads(exc.read().decode("utf-8"))
            if isinstance(body, dict):
                error = body.get("error")
                if isinstance(error, dict):
                    detail = str(error.get("message") or error.get("code") or "")
                detail = detail or str(body.get("detail") or "")
        except Exception:  # noqa: BLE001 - best effort
            detail = ""
        raise RuntimeError(f"HTTP {exc.code}{': ' + detail if detail else ''}") from exc
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"harness API unavailable: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("harness API returned a non-object response")
    return value


def _stream_chat(host: str, payload: dict[str, Any]) -> str:
    request = urllib.request.Request(
        host.rstrip("/") + "/v1/chat/completions",
        data=json.dumps({**payload, "stream": True}).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    chunks: list[str] = []
    try:
        with urllib.request.urlopen(request, timeout=120.0) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if isinstance(event, dict) and event.get("error", {}).get("message"):
                    raise RuntimeError(str(event["error"]["message"]))
                choices = event.get("choices", []) if isinstance(event, dict) else []
                delta = choices[0].get("delta", {}).get("content", "") if choices else ""
                if delta:
                    chunks.append(str(delta))
    except (OSError, urllib.error.URLError, json.JSONDecodeError, IndexError, AttributeError, TypeError) as exc:
        raise RuntimeError(f"harness stream unavailable: {exc}") from exc
    return "".join(chunks)


# --------------------------------------------------------------------- backend
def _api_alive(host: str, *, timeout: float = 1.5) -> bool:
    try:
        _request_json(host, "/healthz", timeout=timeout)
        return True
    except RuntimeError:
        return False


def start_local_backend(*, model_path: str | None, llama_exe: str | None, qlh_base_url: str) -> tuple[Any, str]:
    """Start an in-process harness API; return ``(shell, url)``.

    Reuses the desktop shell's assembly, so there is exactly one place that decides between
    the QLH main-project adapter and a bundled llama-server.
    """

    from .desktop import DesktopShell, DesktopShellConfig

    extra: dict[str, Any] = {}
    if model_path:
        extra["model"] = model_path
    if llama_exe:
        extra["executable"] = llama_exe
    shell = DesktopShell(
        DesktopShellConfig(
            backend="llama" if model_path else "qlh",
            qlh_base_url=qlh_base_url,
            port=0,  # pick a free port
            open_window=False,
            extra=extra,
        )
    )
    server_url = shell.start_server()
    # start_server 返回的是 UI 地址（".../app/"），而 harness API 端点在根路径上。
    return shell, server_url.rstrip("/").removesuffix("/app")


def _probe(host: str) -> dict[str, Any]:
    """Collect the panels' data.  Pure I/O: never touches widgets."""

    data: dict[str, Any] = {}
    try:
        health = _request_json(host, "/healthz")
        data["backend"] = str(health.get("backend", "harness api"))
    except RuntimeError as exc:
        data["error"] = str(exc)
        return data
    try:
        rag = _request_json(host, "/v1/rag/health")
        data["rag"] = f"RAG {rag.get('backend', 'unknown')} / {rag.get('chunks', 0)} chunks"
        data["rag_raw"] = rag
    except RuntimeError:
        data["rag"] = "RAG unavailable"
    try:
        image = _request_json(host, "/v1/images/capabilities")
        ready = bool(image.get("runtime_available")) and bool(image.get("supports_txt2img"))
        data["image"] = "TXT2IMG ready" if ready else "TXT2IMG blocked"
        data["image_raw"] = image
    except RuntimeError:
        data["image"] = "TXT2IMG unavailable"
    try:
        sessions = _request_json(host, "/v1/sessions?owner_scope=local&limit=50")
        data["sessions"] = [item for item in sessions.get("sessions", []) if isinstance(item, dict)]
    except RuntimeError:
        data["sessions"] = []
    try:
        assets = _request_json(host, "/v1/model-assets")
        data["models"] = [item for item in assets.get("models", []) if isinstance(item, dict)]
    except RuntimeError:
        data["models"] = []
    try:
        current = _request_json(host, "/v1/models")
        rows = current.get("data") or []
        if rows and isinstance(rows[0], dict):
            data["loaded_model"] = str(rows[0].get("id", ""))
    except RuntimeError:
        pass
    try:
        tools = _request_json(host, "/v1/mcp/tools")
        data["mcp_tools"] = [item for item in tools.get("tools", []) if isinstance(item, dict)]
    except RuntimeError:
        data["mcp_tools"] = []
    return data


# --------------------------------------------------------------- pure renderers
def format_rag_result(query: str, result: dict[str, Any], *, limit: int, backend: str, chunks: int) -> str:
    """Render one ``/v1/rag/search`` response as plain text (pure, testable)."""

    hits = [hit for hit in (result.get("hits") or []) if isinstance(hit, dict)]
    context = result.get("context") or {}
    lines = [f"知识库（RAG） · 查询「{query}」", ""]
    lines.append(f"索引后端 : {backend}")
    lines.append(f"分块总数 : {chunks}")
    lines.append(f"检索模式 : {result.get('index_mode', 'fts')} · 上限 {limit}")
    lines.append("")
    if not hits:
        lines.append("（没有命中。库为空或关键词不匹配 —— 可用下方「入库」框先添加文档。）")
    else:
        lines.append(f"命中 {len(hits)} 条：")
        for i, hit in enumerate(hits, 1):
            title = str(hit.get("title") or hit.get("source_id") or "(无标题)")
            score = hit.get("score")
            score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "-"
            extra = ""
            if isinstance(hit.get("fusion_score"), (int, float)):
                extra = f" fusion {hit['fusion_score']:.3f}"
            if isinstance(hit.get("rerank_score"), (int, float)):
                extra += f" rerank {hit['rerank_score']:.3f}"
            lines.append("")
            lines.append(f"{i}. {title}   [score {score_text}{extra}]")
            lines.append(f"   {hit.get('source_id', '')} / {hit.get('chunk_id', '')}")
            text = " ".join(str(hit.get("text") or "").split())
            if text:
                lines.append(f"   {text[:HIT_SNIPPET_CHARS]}{'…' if len(text) > HIT_SNIPPET_CHARS else ''}")
    chars = context.get("char_count")
    if not isinstance(chars, int):
        chars = len(str(context.get("text") or ""))
    lines.append("")
    lines.append(f"上下文 : {chars} 字符（由 build_context 截断，供拼进 prompt 用）")
    return "\n".join(lines)


def format_add_result(source_ref: str, title: str, text: str, result: dict[str, Any]) -> str:
    """Render the ``POST /v1/rag/sources`` outcome (pure, testable).

    The response shape is not pinned down here on purpose: the actual keys are listed so a
    contract change is visible in the UI instead of silently rendering nothing.
    """

    lines = [
        "知识库（RAG） · 入库成功",
        "",
        f"来源     : {source_ref}",
        f"标题     : {title}",
        f"字符数   : {len(text)}",
    ]
    if isinstance(result.get("source_id"), str):
        lines.append(f"源 ID    : {result['source_id']}")
    for key in ("chunks", "chunk_count", "added", "indexed"):
        value = result.get(key)
        if isinstance(value, int):
            lines.append(f"{key:<8} : {value}")
    lines.append(f"返回字段 : {', '.join(sorted(str(k) for k in result)) or '(无)'}")
    lines.append("")
    lines.append("现在可在「检索」框输入关键词并回车。")
    return "\n".join(lines)


def tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """Return a tool's input schema, tolerating either naming convention."""

    schema = tool.get("input_schema")
    if not isinstance(schema, dict):
        schema = tool.get("inputSchema")
    return schema if isinstance(schema, dict) else {}


def mcp_arguments_template(schema: dict[str, Any] | None) -> str:
    """Build a JSON skeleton from a tool input schema (pure, testable).

    只保留必填字段：否则一屏全是空值，用户还得先删。
    """

    spec = schema if isinstance(schema, dict) else {}
    properties = spec.get("properties")
    properties = properties if isinstance(properties, dict) else {}
    required = [key for key in (spec.get("required") or []) if isinstance(key, str)]
    keys = required or [key for key in properties if isinstance(key, str)]
    skeleton: dict[str, Any] = {}
    for key in keys:
        info = properties.get(key)
        kind = info.get("type") if isinstance(info, dict) else None
        if key == "owner_scope":
            skeleton[key] = "local"
        elif kind == "string":
            skeleton[key] = ""
        elif kind in ("integer", "number"):
            skeleton[key] = 0
        elif kind == "boolean":
            skeleton[key] = False
        elif kind == "array":
            skeleton[key] = []
        elif kind == "object":
            skeleton[key] = {}
        else:
            skeleton[key] = None
    return json.dumps(skeleton, ensure_ascii=False, indent=2)


def format_mcp_result(name: str, response: dict[str, Any]) -> str:
    """Render a JSON-RPC ``tools/call`` response (pure, testable)."""

    lines = [f"MCP · {name}", ""]
    error = response.get("error")
    if isinstance(error, dict):
        lines.append(f"调用失败 : {error.get('message') or error.get('code') or error}")
        return "\n".join(lines)
    result = response.get("result")
    if not isinstance(result, dict):
        lines.append("（响应里没有 result 字段）")
        lines.append(f"原始响应 : {json.dumps(response, ensure_ascii=False)[:400]}")
        return "\n".join(lines)
    if result.get("isError"):
        lines.append("工具报告错误 :")
    content = result.get("content")
    if isinstance(content, list) and content:
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                lines.append(str(text) if text is not None else json.dumps(block, ensure_ascii=False))
            else:
                lines.append(str(block))
    else:
        lines.append(json.dumps(result, ensure_ascii=False, indent=2)[:1200])
    structured = result.get("structuredContent")
    if isinstance(structured, dict) and structured:
        lines.append("")
        lines.append("结构化结果 :")
        lines.append(json.dumps(structured, ensure_ascii=False, indent=2)[:1200])
    return "\n".join(lines)


def parse_mcp_arguments(raw: str) -> dict[str, Any]:
    """Parse the arguments box; raise ``ValueError`` with a readable reason."""

    text = raw.strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"参数不是合法 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("参数必须是 JSON 对象（例如 {\"query\": \"缓存\"}）")
    return value


def parse_image_size(raw: str) -> tuple[int, int]:
    """Parse ``WxH`` into ``(width, height)``; raise ``ValueError`` when unusable."""

    text = raw.strip().lower().replace("*", "x").replace("×", "x")
    if "x" not in text:
        raise ValueError(f"尺寸格式应为 WxH（如 512x512），收到: {raw!r}")
    left, _, right = text.partition("x")
    try:
        width, height = int(left), int(right)
    except ValueError as exc:
        raise ValueError(f"尺寸必须是整数: {raw!r}") from exc
    if not (64 <= width <= 2048 and 64 <= height <= 2048):
        raise ValueError(f"尺寸需在 64..2048 之间，收到 {width}x{height}")
    return width, height


def image_output_dir(explicit: str | None = None) -> Path:
    """Where generated images land: explicit dir, else the shared Koakumix data dir."""

    if explicit:
        return Path(explicit).expanduser()
    try:
        from .desktop import default_data_dir

        base = default_data_dir()
    except Exception:  # pragma: no cover - desktop module always importable here
        base = Path.home() / ".koakumix"
    return Path(base) / "images"


def save_generated_image(item: dict[str, Any], out_dir: Path, *, stamp: str | None = None) -> Path:
    """Decode a ``b64_json`` payload and write it under ``out_dir``; return the path.

    A terminal cannot show a picture, so the only useful thing to do with one is put it on
    disk and report the path.
    """

    raw = item.get("b64_json")
    if not isinstance(raw, str) or not raw:
        raise ValueError("响应里没有 b64_json（可用 response_format=url 改走资产库）")
    try:
        blob = base64.b64decode(raw, validate=True)
    except Exception as exc:  # noqa: BLE001 - binascii.Error and friends
        raise ValueError(f"b64_json 解码失败: {exc}") from exc
    if not blob:
        raise ValueError("解码后得到空数据")

    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    mime = str(metadata.get("mime_type") or "image/png")
    suffix = MIME_SUFFIXES.get(mime, ".png")
    seed = metadata.get("seed")
    tag = f"-seed{seed}" if isinstance(seed, int) else ""
    name = f"koakumix-{stamp or _time.strftime('%Y%m%d-%H%M%S')}{tag}{suffix}"

    target_dir = Path(out_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / name
    path.write_bytes(blob)
    return path


def format_image_result(path: Path, item: dict[str, Any], *, prompt: str = "") -> str:
    """Render the generation outcome (pure, testable)."""

    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    lines = ["资产 · 图像生成", ""]
    if prompt:
        lines.append(f"提示词 : {prompt[:160]}{'…' if len(prompt) > 160 else ''}")
    lines.append(f"已保存 : {path}")
    lines.append(f"字节数 : {path.stat().st_size if path.exists() else 0}")
    if isinstance(item.get("asset_id"), str):
        lines.append(f"资产 ID : {item['asset_id']}")
    for key in ("width", "height", "steps", "guidance_scale", "seed", "mime_type", "backend_id"):
        value = metadata.get(key)
        if value is not None:
            lines.append(f"{key:<14}: {value}")
    if metadata:
        listed = ", ".join(sorted(str(k) for k in metadata))
        lines.append(f"元数据字段    : {listed}")
    lines.append("")
    lines.append("终端无法内嵌显示图片，请用系统图片查看器打开上面的路径。")
    return "\n".join(lines)


def format_profile_list(profiles: list[dict[str, Any]], *, limit: int = 10) -> str:
    """One line per model profile (pure, testable)."""

    if not profiles:
        return "（没有画像）"
    lines = []
    for profile in profiles[:limit]:
        model_id = str(profile.get("model_id") or "?")
        backend = str(profile.get("backend") or "-")
        fmt = str(profile.get("format") or "-")
        lines.append(f"{model_id}  [{backend}/{fmt}]")
    if len(profiles) > limit:
        lines.append(f"… 另有 {len(profiles) - limit} 个")
    return "\n".join(lines)


def format_model_library(
    profiles: list[dict[str, Any]],
    presets: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    *,
    selected: str = "",
    job_limit: int = 5,
) -> str:
    """Render the model library panel (pure, testable).

    字段名取自实测响应（profiles: model_id/backend/format；presets: id/display/kind/
    installable/blocked_reasons；jobs: status/progress/downloaded_bytes/error）。
    """

    lines = ["模型库", ""]
    lines.append(f"画像     : {len(profiles)} 个")
    lines.append(f"预设     : {len(presets)} 个")
    lines.append(f"下载队列 : {len(jobs)} 项")

    if profiles:
        lines.append("")
        lines.append("画像一览：")
        lines.append(format_profile_list(profiles))

    if selected:
        preset = next((p for p in presets if str(p.get("id")) == selected), None)
        lines.append("")
        lines.append(f"选中预设 : {selected}")
        if isinstance(preset, dict):
            display = preset.get("display")
            if display:
                lines.append(f"  名称     : {display}")
            lines.append(f"  类型     : {preset.get('kind', '-')}")
            lines.append(f"  默认引擎 : {preset.get('default_engine', '-')} / {preset.get('default_quant', '-')}")
            lines.append(f"  可安装   : {bool(preset.get('installable'))}")
            blocked = preset.get("blocked_reasons")
            if isinstance(blocked, list) and blocked:
                lines.append(f"  阻塞原因 : {'; '.join(str(item) for item in blocked)}")
            description = preset.get("description")
            if description:
                lines.append(f"  说明     : {str(description)[:120]}")

    if jobs:
        shown = jobs[-job_limit:]
        lines.append("")
        lines.append(f"最近下载（{len(shown)}/{len(jobs)}）：")
        for job in shown:
            progress = job.get("progress")
            if isinstance(progress, (int, float)):
                pct = f"{float(progress) * 100:>3.0f}%"
            else:
                done, total = job.get("downloaded_bytes"), job.get("total_bytes")
                pct = f"{done}/{total}" if isinstance(done, int) and isinstance(total, int) else "-"
            name = job.get("preset_id") or job.get("model_id") or job.get("job_id") or "?"
            lines.append(f"  {str(job.get('status', '-')):<10} {pct:>7}  {name}")
            if job.get("error"):
                lines.append(f"        错误: {str(job['error'])[:100]}")

    lines.append("")
    lines.append("在下方列表里选中一个预设 → 按 d 入队下载（POST /v1/model-downloads）。按 t 刷新。")
    return "\n".join(lines)


def read_source_file(target: str) -> tuple[str, str]:
    """Read a text file for ingestion; return ``(title, text)``.

    Raises ``ValueError`` with an actionable message so the panel can show it verbatim.
    """

    path = Path(target).expanduser()
    if not path.is_file():
        raise ValueError(f"找不到文件: {path}")
    size = path.stat().st_size
    if size > MAX_SOURCE_BYTES:
        raise ValueError(f"文件过大: {size} 字节（上限 {MAX_SOURCE_BYTES // 1024} KiB）")
    if size == 0:
        raise ValueError(f"文件为空: {path}")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ValueError(f"读取失败: {exc}") from exc
    return path.name, text


def create_app(
    *,
    host: str = DEFAULT_HOST,
    model: str | None = None,
    llama_exe: str | None = None,
    serve: bool = True,
    splash: bool = True,
    splash_min: float = 1.0,
    qlh_base_url: str = QLH_BASE_URL,
    rag_limit: int = 8,
    rag_mode: str = "fts",
    image_dir: str | None = None,
    image_size: str = DEFAULT_IMAGE_SIZE,
    image_steps: int = DEFAULT_IMAGE_STEPS,
) -> Any:
    try:
        from textual.app import App, ComposeResult
        from textual.containers import Horizontal, Vertical, VerticalScroll
        from textual.widgets import Button, Footer, Header, Input, Label, ListItem, ListView, Static

        from .splash import SplashScreen
    except ImportError as exc:  # pragma: no cover - optional UI dependency
        raise RuntimeError("Textual is required for the TUI; install the optional harness UI dependency") from exc

    class HarnessApp(App[None]):
        TITLE = "KOAKUMIX"
        SUB_TITLE = "Evangelium vom Himmelsturz."
        CSS = """
        /* 主界面：中性深色（参照 reasonix cli / GitHub Dark）；红色只留给启动页 */
        Screen { background: #0d1117; color: #c9d1d9; }
        Header { background: #161b22; color: #c9d1d9; }
        Footer { background: #161b22; color: #8b949e; }
        #layout { height: 1fr; }
        #rail { width: 32; padding: 1 2; background: #161b22; border-right: solid #30363d; }
        #main { width: 1fr; padding: 1 3; }
        #status { height: 1; color: #58a6ff; }
        #utility-status { height: auto; color: #8b949e; padding: 1 0; }
        #banner { color: #bc8cff; height: auto; padding: 0 0 1 0; }
        #transcript-scroll { height: 1fr; border-top: solid #30363d; }
        #transcript { height: auto; padding: 1 0; }
        #panel { display: none; height: 1fr; border-top: solid #30363d; }
        #panel-scroll { height: 1fr; }
        #panel-text { height: auto; padding: 1 0; }
        #mcp-tool-list { display: none; height: auto; max-height: 8; }
        #preset-list { display: none; height: auto; max-height: 8; }
        #rag-query { display: none; border: solid #30363d; background: #0d1117; }
        #rag-add { display: none; border: solid #30363d; background: #0d1117; }
        #mcp-args { display: none; border: solid #30363d; background: #0d1117; }
        #image-prompt { display: none; border: solid #30363d; background: #0d1117; }
        #composer { dock: bottom; height: 3; border: solid #30363d; background: #0d1117; }
        ListView { height: auto; max-height: 8; background: #161b22; }
        ListItem { padding: 0 1; color: #8b949e; background: #161b22; }
        ListItem:hover { background: #1f2937; color: #c9d1d9; }
        ListItem:focus { background: #1f2937; color: #58a6ff; }
        Input { background: #0d1117; color: #c9d1d9; }
        Input:focus { border: solid #58a6ff; }
        Button { background: #21262d; color: #c9d1d9; border: solid #30363d; }
        Button:focus { background: #30363d; border: solid #58a6ff; }
        .label { color: #8b949e; text-style: bold; padding: 1 0 0 0; }
        .muted { color: #6e7681; }
        """

        BINDINGS = [
            ("1", "nav(0)", "对话"),
            ("2", "nav(1)", "知识库"),
            ("3", "nav(2)", "MCP"),
            ("4", "nav(3)", "资产"),
            ("5", "nav(4)", "模型库"),
            ("6", "nav(5)", "运行时"),
            ("m", "reload_models", "刷新模型"),
            ("t", "refresh", "刷新本页"),
            ("d", "queue_download", "入队下载"),
            ("n", "new_session", "新建会话"),
            ("q", "quit", "退出"),
        ]

        def __init__(self) -> None:
            super().__init__()
            self._serve = bool(serve)
            self._splash = bool(splash)
            self._splash_min = float(splash_min)
            self._host = host
            self._session_id: str | None = None
            self._sessions: list[dict[str, Any]] = []
            self._models: list[dict[str, Any]] = []
            self._current_model: str | None = None
            self._shell: Any | None = None
            self._nav = 0
            self._boot: dict[str, Any] = {}
            self._rag_limit = int(rag_limit)
            self._rag_mode = str(rag_mode)
            self._last_query = ""
            self._last_source = ""
            self._mcp_tools: list[dict[str, Any]] = []
            self._mcp_tool: str = ""
            self._image_dir = image_dir
            self._image_size = str(image_size)
            self._image_steps = int(image_steps)
            self._last_image: str = ""
            self._profiles: list[dict[str, Any]] = []
            self._presets: list[dict[str, Any]] = []
            self._jobs: list[dict[str, Any]] = []
            self._preset: str = ""

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal(id="layout"):
                with Vertical(id="rail"):
                    yield Label("WORKSPACE", classes="label")
                    yield ListView(
                        *[
                            ListItem(Label(f"{'▸' if i == 0 else ' '} {title}"), name=key)
                            for i, (key, title) in enumerate(NAV_ITEMS)
                        ],
                        id="nav-list",
                    )
                    yield Label("MODEL", classes="label")
                    yield ListView(id="model-list")
                    yield Label("SESSION", classes="label")
                    yield Button("+ 新建会话", id="new-session")
                    yield ListView(id="session-list")
                    yield Label("RAG / ASSETS", classes="label")
                    yield Static("等待能力探测", id="utility-status", classes="muted")
                with Vertical(id="main"):
                    yield Static("CHECKING · harness API", id="status")
                    yield Static("Evangelium vom Himmelsturz.", id="banner")
                    with VerticalScroll(id="transcript-scroll"):
                        yield Static("KOAKUMIX\n\n等待一条消息。", id="transcript")
                    with Vertical(id="panel"):
                        with VerticalScroll(id="panel-scroll"):
                            yield Static("", id="panel-text")
                        yield ListView(id="mcp-tool-list")
                        yield ListView(id="preset-list")
                        yield Input(placeholder="检索知识库…（回车执行）", id="rag-query")
                        yield Input(placeholder="入库：本地文档路径…（回车读取并索引）", id="rag-add")
                        yield Input(placeholder="MCP 参数（JSON）…（回车调用）", id="mcp-args")
                        yield Input(placeholder="图像提示词…（回车生成并保存到本地）", id="image-prompt")
                    yield Input(placeholder="输入消息，回车发送…", id="composer")
            yield Footer()

        def on_mount(self) -> None:
            if self._splash and not self.is_headless:
                self.push_screen(SplashScreen("少女祈祷中……", min_show=self._splash_min))
                self.run_worker(self._boot_worker, thread=True, name="boot")
            else:
                self._boot_worker()

        def on_unmount(self) -> None:
            shell = self._shell
            if shell is not None:
                try:
                    shell.shutdown()
                except Exception:  # noqa: BLE001 - best effort on exit
                    pass

        # ---- boot：先复用、再自起；网络全在 worker 线程 ----
        def _boot_worker(self) -> None:
            note = "reused"
            if not _api_alive(self._host):
                if self._serve:
                    try:
                        shell, url = start_local_backend(
                            model_path=model, llama_exe=llama_exe, qlh_base_url=qlh_base_url
                        )
                        self._shell = shell
                        self._host = url
                        note = "started"
                    except Exception as exc:  # noqa: BLE001 - degrade, never crash the UI
                        note = f"start_failed: {exc}"
                else:
                    note = "serve_disabled"
            data = _probe(self._host)
            library = _fetch_model_library(self._host)
            data.update(library)
            self._deliver(note, data)

        def _deliver(self, note: str, data: dict[str, Any]) -> None:
            """Safe from a worker thread *or* the app thread (headless path)."""

            if threading.current_thread() is threading.main_thread():
                self._apply_boot(note, data)
            else:
                self.call_from_thread(self._apply_boot, note, data)

        def _apply_boot(self, note: str, data: dict[str, Any]) -> None:
            status = self.query_one("#status", Static)
            utility = self.query_one("#utility-status", Static)
            self._boot = data
            if data.get("error"):
                status.update("OFFLINE · 后端未连接（发送消息只会记录输入）")
                utility.update(
                    f"RAG / ASSETS · unavailable\n原因: {note}\n"
                    "提示: 加 --model <路径.gguf>，或先起主项目 API / harness API"
                )
            else:
                where = {"reused": "已复用", "started": "已自动拉起"}.get(note, note)
                status.update(f"ONLINE · {data.get('backend', 'harness api')} · {where}")
                utility.update(f"{data.get('rag', 'RAG unknown')}\n{data.get('image', 'TXT2IMG unknown')}")
                self._sessions = list(data.get("sessions") or [])
                self._render_sessions()
                self._models = list(data.get("models") or [])
                self._current_model = data.get("loaded_model") or None
                self._render_models()
                self._mcp_tools = list(data.get("mcp_tools") or [])
                self._render_mcp_tools()
                self._profiles = list(data.get("profiles") or [])
                self._presets = list(data.get("presets") or [])
                self._jobs = list(data.get("jobs") or [])
                self._render_presets()
                if self._sessions:
                    self._select_session(str(self._sessions[0].get("session_id", "")))
            self._render_nav()
            self._close_splash()

        def _close_splash(self) -> None:
            screen = self.screen
            if isinstance(screen, SplashScreen):
                try:
                    screen.notify_loaded()
                except Exception:  # noqa: BLE001 - screen already gone
                    pass

        # ---- transcript：唯一写入口，写完滚到底 ----
        def _set_transcript(self, text: str) -> None:
            """Single writer for the transcript, keeping the newest line in view.

            Without the explicit scroll the view stays at the top, so a long session looks
            like "the model never answered" — the reply is simply below the fold.
            """

            self.query_one("#transcript", Static).update(text)
            self._scroll("#transcript-scroll")

        def _transcript_text(self) -> str:
            # Textual 8.x 的 Static 没有 .renderable；正文用官方属性 .content 读回。
            return str(self.query_one("#transcript", Static).content or "")

        def _scroll(self, selector: str) -> None:
            try:
                self.query_one(selector).scroll_end(animate=False)
            except Exception:  # noqa: BLE001 - container not mounted yet
                pass

        def _set_panel(self, text: str) -> None:
            self.query_one("#panel-text", Static).update(text)
            self._scroll("#panel-scroll")

        def _refresh_rag_health(self) -> None:
            """Re-read /v1/rag/health so the chunk count reflects what we just ingested."""

            try:
                rag = _request_json(self._host, "/v1/rag/health")
            except RuntimeError:
                return
            self._boot["rag_raw"] = rag
            self._boot["rag"] = f"RAG {rag.get('backend', 'unknown')} / {rag.get('chunks', 0)} chunks"
            try:
                self.query_one("#utility-status", Static).update(
                    f"{self._boot['rag']}\n{self._boot.get('image', 'TXT2IMG unknown')}"
                )
            except Exception:  # noqa: BLE001
                pass

        # ---- navigation ----
        def action_nav(self, index: int) -> None:
            self._nav = max(0, min(index, len(NAV_ITEMS) - 1))
            self._render_nav()

        def _render_nav(self) -> None:
            """Switch the main area between the chat view and the info panels."""

            chatting = self._nav == 0
            on_library = self._nav == 1
            on_mcp = self._nav == 2
            on_assets = self._nav == 3
            on_presets = self._nav == 4
            self.query_one("#transcript-scroll").display = chatting
            self.query_one("#composer").display = chatting
            self.query_one("#panel").display = not chatting
            # 各页专属控件：只在自己的页面上出现。
            self.query_one("#rag-query").display = on_library
            self.query_one("#rag-add").display = on_library
            self.query_one("#mcp-tool-list").display = on_mcp
            self.query_one("#mcp-args").display = on_mcp
            self.query_one("#image-prompt").display = on_assets
            self.query_one("#preset-list").display = on_presets
            self._mark_nav()
            if chatting:
                return
            builders = (
                self._library_panel,
                self._mcp_panel,
                self._assets_panel,
                self._presets_panel,
                self._runtime_panel,
            )
            self._set_panel(builders[self._nav - 1]())

        def _mark_nav(self) -> None:
            view = self.query_one("#nav-list", ListView)
            for i, (key, title) in enumerate(NAV_ITEMS):
                if i < len(view.children):
                    view.children[i].query_one(Label).update(f"{'▸' if i == self._nav else ' '} {title}")

        def _library_panel(self) -> str:
            rag = self._boot.get("rag_raw") or {}
            head = (
                "知识库（RAG）\n\n"
                f"索引后端 : {rag.get('backend', 'unavailable')}\n"
                f"分块总数 : {rag.get('chunks', 0)}\n"
                f"检索模式 : {self._rag_mode} · 上限 {self._rag_limit}\n"
            )
            if self._last_source:
                head += f"最近入库 : {self._last_source}\n"
            return (
                head + "\n"
                "· 上面第一个框：关键词检索（POST /v1/rag/search）\n"
                "· 上面第二个框：填入本地文档路径回车，读入并索引（POST /v1/rag/sources）"
            )

        def _mcp_panel(self) -> str:
            if not self._mcp_tools:
                return (
                    "MCP 工具\n\n"
                    "（没有取到工具列表。按 t 重新拉取；后端未连接时列表为空。）\n\n"
                    "工具列表来自 GET /v1/mcp/tools，调用走 POST /v1/mcp/call。"
                )
            selected = self._mcp_tool or "(未选中)"
            lines = [f"MCP 工具 · 共 {len(self._mcp_tools)} 个", ""]
            lines.append(f"当前选中 : {selected}")
            lines.append("")
            lines.append("在下方列表里选中一个工具 → 参数框会自动填入必填字段的 JSON 骨架 →")
            lines.append("填好值后回车调用。按 t 可重新拉取工具列表。")
            return "\n".join(lines)

        def _assets_panel(self) -> str:
            image = self._boot.get("image_raw") or {}
            rows = [f"已注册模型 : {len(self._models)}"]
            if self._current_model:
                rows.append(f"当前模型   : {self._current_model}")
            rows.append(f"图像运行时 : {self._boot.get('image', 'unavailable')}")
            rows.append(f"MCP 工具数 : {len(self._mcp_tools)}")
            rows.append(f"生成尺寸   : {self._image_size} · 步数 {self._image_steps}")
            rows.append(f"输出目录   : {image_output_dir(self._image_dir)}")
            if image:
                rows.append(f"  txt2img  : {bool(image.get('supports_txt2img'))}")
                rows.append(f"  runtime  : {bool(image.get('runtime_available'))}")
            if self._last_image:
                rows.append(f"最近生成   : {self._last_image}")
            rows.append("")
            rows.append("在下方提示词框输入描述并回车即可生成（POST /v1/images/generations），")
            rows.append("图片会保存到上面的输出目录。")
            return "资产\n\n" + "\n".join(rows)

        def _presets_panel(self) -> str:
            return format_model_library(
                self._profiles,
                self._presets,
                self._jobs,
                selected=self._preset,
            )

        def _runtime_panel(self) -> str:
            return (
                "运行时\n\n"
                f"后端地址 : {self._host}\n"
                f"后端类型 : {self._boot.get('backend', 'unavailable')}\n"
                f"内嵌后端 : {'是（本 TUI 拉起）' if self._shell is not None else '否（复用外部服务）'}\n"
                f"会话数   : {len(self._sessions)}\n"
                f"模型数   : {len(self._models)}\n"
                f"MCP 工具 : {len(self._mcp_tools)}\n"
                f"画像数   : {len(self._profiles)}\n"
                f"预设数   : {len(self._presets)}\n"
                f"图像输出 : {image_output_dir(self._image_dir)}"
            )

        # ---- knowledge base ----
        def _run_rag_search(self, query: str) -> None:
            if not query:
                self._set_panel(self._library_panel())
                return
            status = self.query_one("#status", Static)
            status.update(f"SEARCHING · {query}")
            self._set_panel(f"知识库（RAG） · 查询「{query}」\n\n检索中…")
            try:
                result = _request_json(
                    self._host,
                    "/v1/rag/search",
                    {
                        "query": query,
                        "owner_scope": "local",
                        "limit": self._rag_limit,
                        "index_mode": self._rag_mode,
                    },
                    timeout=60.0,
                )
            except RuntimeError as exc:
                self._set_panel(f"知识库（RAG） · 查询「{query}」\n\n检索失败: {exc}")
                status.update("OFFLINE · 知识库检索失败")
                return
            rag = self._boot.get("rag_raw") or {}
            self._last_query = query
            self._set_panel(
                format_rag_result(
                    query,
                    result,
                    limit=self._rag_limit,
                    backend=str(rag.get("backend", "unavailable")),
                    chunks=int(rag.get("chunks", 0) or 0),
                )
            )
            hits = len([h for h in (result.get("hits") or []) if isinstance(h, dict)])
            status.update(f"ONLINE · 知识库命中 {hits} 条")

        def _add_rag_source(self, target: str) -> None:
            if not target:
                self._set_panel(self._library_panel())
                return
            status = self.query_one("#status", Static)
            try:
                title, text = read_source_file(target)
            except ValueError as exc:
                self._set_panel(f"知识库（RAG） · 入库\n\n入库失败：{exc}")
                status.update("OFFLINE · 入库失败（文件不可用）")
                return
            status.update(f"INDEXING · {title}")
            self._set_panel(f"知识库（RAG） · 入库\n\n正在索引 {title} …")
            try:
                result = _request_json(
                    self._host,
                    "/v1/rag/sources",
                    {
                        "source_ref": str(Path(target).expanduser()),
                        "title": title,
                        "text": text,
                        "owner_scope": "local",
                    },
                    timeout=120.0,
                )
            except RuntimeError as exc:
                self._set_panel(f"知识库（RAG） · 入库\n\n入库失败：{exc}")
                status.update("OFFLINE · 入库失败")
                return
            self._last_source = title
            self._refresh_rag_health()
            self._set_panel(format_add_result(str(Path(target).expanduser()), title, text, result))
            status.update(f"ONLINE · 已入库 {title}")

        # ---- assets：图像生成 ----
        def _generate_image(self, prompt: str) -> None:
            if not prompt:
                self._set_panel(self._assets_panel())
                return
            status = self.query_one("#status", Static)
            try:
                width, height = parse_image_size(self._image_size)
            except ValueError as exc:
                self._set_panel(f"资产 · 图像生成\n\n参数错误：{exc}")
                status.update("OFFLINE · 图像尺寸参数错误")
                return
            status.update("GENERATING · 图像生成中（可能几十秒）")
            self._set_panel(f"资产 · 图像生成\n\n提示词：{prompt[:120]}\n\n生成中…")
            try:
                payload = _request_json(
                    self._host,
                    "/v1/images/generations",
                    {
                        "prompt": prompt,
                        "width": width,
                        "height": height,
                        "steps": self._image_steps,
                        "response_format": "b64_json",
                    },
                    timeout=300.0,
                )
            except RuntimeError as exc:
                self._set_panel(f"资产 · 图像生成\n\n生成失败：{exc}")
                status.update("OFFLINE · 图像生成失败")
                return

            rows = payload.get("data")
            item = rows[0] if isinstance(rows, list) and rows and isinstance(rows[0], dict) else None
            if item is None:
                self._set_panel("资产 · 图像生成\n\n响应里没有 data[0]，无法取图。")
                status.update("OFFLINE · 图像响应异常")
                return
            try:
                path = save_generated_image(item, image_output_dir(self._image_dir))
            except ValueError as exc:
                self._set_panel(f"资产 · 图像生成\n\n保存失败：{exc}")
                status.update("OFFLINE · 图像保存失败")
                return
            self._last_image = str(path)
            self._set_panel(format_image_result(path, item, prompt=prompt))
            status.update(f"ONLINE · 已生成 {path.name}")

        # ---- model library ----
        def _render_presets(self) -> None:
            view = self.query_one("#preset-list", ListView)
            view.clear()
            for preset in self._presets:
                preset_id = str(preset.get("id") or "")
                if not preset_id:
                    continue
                mark = "▸ " if preset_id == self._preset else "  "
                label = str(preset.get("display") or preset_id)
                if not preset.get("installable", True):
                    label += "  (不可安装)"
                view.append(ListItem(Label(f"{mark}{label}"), name=preset_id))

        def _select_preset(self, preset_id: str) -> None:
            if not preset_id:
                return
            self._preset = preset_id
            self._render_presets()
            if self._nav == 4:
                self._set_panel(self._presets_panel())

        def _refresh_model_library(self) -> bool:
            """Reload the library; return whether any endpoint actually answered."""

            library = _fetch_model_library(self._host)
            self._profiles = list(library.get("profiles") or [])
            self._presets = list(library.get("presets") or [])
            self._jobs = list(library.get("jobs") or [])
            self._render_presets()
            if self._nav == 4:
                self._set_panel(self._presets_panel())
            return bool(library.get("ok"))

        def action_queue_download(self) -> None:
            """Queue the selected preset.  An explicit key, because this has side effects."""

            status = self.query_one("#status", Static)
            preset_id = self._preset
            if not preset_id:
                status.update("OFFLINE · 未选中预设（先在下方列表里选一个）")
                return
            status.update(f"QUEUING · {preset_id}")
            try:
                _request_json(self._host, "/v1/model-downloads", {"preset_id": preset_id}, timeout=60.0)
            except RuntimeError as exc:
                status.update(f"OFFLINE · 入队失败: {exc}")
                self._set_panel(f"模型库 · 入队失败\n\n{preset_id}\n\n{exc}")
                return
            self._refresh_model_library()
            status.update(f"ONLINE · 已入队 {preset_id}")

        # ---- MCP ----
        def _render_mcp_tools(self) -> None:
            view = self.query_one("#mcp-tool-list", ListView)
            view.clear()
            for tool in self._mcp_tools:
                name = str(tool.get("name") or "")
                if not name:
                    continue
                mark = "▸ " if name == self._mcp_tool else "  "
                view.append(ListItem(Label(f"{mark}{name}"), name=name))

        def _reload_mcp_tools(self) -> None:
            try:
                payload = _request_json(self._host, "/v1/mcp/tools")
            except RuntimeError:
                self.query_one("#status", Static).update("OFFLINE · MCP 工具列表不可用")
                return
            self._mcp_tools = [item for item in payload.get("tools", []) if isinstance(item, dict)]
            self._render_mcp_tools()
            if self._nav == 2:
                self._render_nav()
            self.query_one("#status", Static).update(f"ONLINE · MCP 工具 {len(self._mcp_tools)} 个")

        def action_refresh(self) -> None:
            """Refresh whatever the current page pulled from the API."""

            status = self.query_one("#status", Static)
            if self._nav == 2:
                self._reload_mcp_tools()
            elif self._nav == 4:
                ok = self._refresh_model_library()
                status.update("ONLINE · 模型库已刷新" if ok else "OFFLINE · 模型库不可用（后端未连接）")
            else:
                status.update("（本页没有需要刷新的数据）")

        def _select_mcp_tool(self, name: str) -> None:
            if not name:
                return
            self._mcp_tool = name
            tool = next((t for t in self._mcp_tools if str(t.get("name")) == name), None)
            description = str((tool or {}).get("description") or "")
            template = mcp_arguments_template(tool_schema(tool or {}))
            try:
                self.query_one("#mcp-args", Input).value = "" if template.strip() == "{}" else template
                self.query_one("#mcp-args", Input).focus()
            except Exception:  # noqa: BLE001
                pass
            self._render_mcp_tools()
            self._set_panel(
                "\n".join(
                    [
                        f"MCP · {name}",
                        "",
                        description or "(无描述)",
                        "",
                        "参数已填入下方输入框（只含必填字段）。补齐后回车调用。",
                    ]
                )
            )

        def _call_mcp(self, raw: str) -> None:
            name = self._mcp_tool
            if not name:
                self._set_panel("MCP · 尚未选中工具\n\n请先在下方列表里选一个工具。")
                return
            status = self.query_one("#status", Static)
            try:
                arguments = parse_mcp_arguments(raw)
            except ValueError as exc:
                self._set_panel(f"MCP · {name}\n\n参数错误：{exc}")
                status.update("OFFLINE · MCP 参数错误")
                return
            status.update(f"CALLING · {name}")
            self._set_panel(f"MCP · {name}\n\n调用中…")
            try:
                response = _request_json(
                    self._host,
                    "/v1/mcp/call",
                    {"name": name, "arguments": arguments},
                    timeout=120.0,
                )
            except RuntimeError as exc:
                self._set_panel(f"MCP · {name}\n\n调用失败：{exc}")
                status.update("OFFLINE · MCP 调用失败")
                return
            self._set_panel(format_mcp_result(name, response))
            failed = bool(response.get("error")) or bool((response.get("result") or {}).get("isError"))
            status.update(f"{'OFFLINE' if failed else 'ONLINE'} · MCP {name} {'失败' if failed else '完成'}")

        # ---- models（左栏快捷切换）----
        def _render_models(self) -> None:
            view = self.query_one("#model-list", ListView)
            view.clear()
            for item in self._models:
                model_id = str(item.get("model_id") or item.get("id") or "")
                if not model_id:
                    continue
                mark = "▸ " if model_id == self._current_model else "  "
                label = str(item.get("name") or model_id)
                view.append(ListItem(Label(f"{mark}{label}"), name=model_id))

        def action_reload_models(self) -> None:
            try:
                assets = _request_json(self._host, "/v1/model-assets")
            except RuntimeError:
                self.query_one("#status", Static).update("OFFLINE · 模型列表不可用")
                return
            self._models = [item for item in assets.get("models", []) if isinstance(item, dict)]
            self._render_models()
            if self._nav == 3:
                self._render_nav()

        def _switch_model(self, model_id: str) -> None:
            if not model_id:
                return
            status = self.query_one("#status", Static)
            status.update(f"LOADING · {model_id} …（首次加载可能十几秒）")
            try:
                _request_json(
                    self._host, "/v1/models/load", {"model_id": model_id, "engine": "auto"}, timeout=300.0
                )
            except RuntimeError as exc:
                status.update(f"MODEL FAILED · {exc}")
                return
            self._current_model = model_id
            self._render_models()
            status.update(f"ONLINE · model={model_id}")

        # ---- sessions ----
        def _render_sessions(self) -> None:
            session_list = self.query_one("#session-list", ListView)
            session_list.clear()
            for item in self._sessions:
                session_list.append(
                    ListItem(Label(str(item.get("title", "New session"))), name=str(item.get("session_id", "")))
                )

        def _select_session(self, session_id: str) -> None:
            if not session_id:
                return
            try:
                payload = _request_json(self._host, f"/v1/sessions/{session_id}?owner_scope=local")
            except RuntimeError:
                self.query_one("#status", Static).update("OFFLINE · 会话加载失败")
                return
            self._session_id = session_id
            lines = []
            for message in payload.get("messages", []):
                role = str(message.get("role", "system")).upper()
                lines.append(f"{role}\n{message.get('content', '')}")
            self._set_transcript("\n\n".join(lines) or "KOAKUMIX\n\n等待一条消息。")

        def on_list_view_selected(self, event: Any) -> None:
            which = getattr(event.list_view, "id", None)
            name = str(getattr(event.item, "name", ""))
            if which == "nav-list":
                index = next((i for i, (key, _) in enumerate(NAV_ITEMS) if key == name), 0)
                self.action_nav(index)
            elif which == "session-list":
                self._select_session(name)
            elif which == "model-list":
                self._switch_model(name)
            elif which == "mcp-tool-list":
                self._select_mcp_tool(name)
            elif which == "preset-list":
                self._select_preset(name)

        def action_new_session(self) -> None:
            self._create_session()

        def on_button_pressed(self, event: Any) -> None:
            if getattr(event.button, "id", None) == "new-session":
                self._create_session()

        def _create_session(self) -> None:
            try:
                created = _request_json(self._host, "/v1/sessions", {"owner_scope": "local", "title": "New session"})
            except RuntimeError:
                self.query_one("#status", Static).update("OFFLINE · 无法新建会话（后端未连接）")
                return
            self._session_id = str(created.get("session_id", ""))
            try:
                payload = _request_json(self._host, "/v1/sessions?owner_scope=local&limit=50")
                self._sessions = [item for item in payload.get("sessions", []) if isinstance(item, dict)]
            except RuntimeError:
                self._sessions = []
            self._render_sessions()
            if self._nav != 0:
                self.action_nav(0)
            self._set_transcript("KOAKUMIX\n\n等待一条消息。")

        # ---- inputs：检索 / 入库 / MCP 参数 / 图像提示词 / 对话 分流 ----
        def on_input_submitted(self, event: Input.Submitted) -> None:
            widget_id = getattr(event.input, "id", None)
            text = event.value.strip()
            if widget_id == "rag-query":
                event.input.value = ""
                self._run_rag_search(text)
                return
            if widget_id == "rag-add":
                event.input.value = ""
                self._add_rag_source(text)
                return
            if widget_id == "mcp-args":
                # 参数框保留原值：允许在同一次会话里微调后重复调用。
                self._call_mcp(event.value)
                return
            if widget_id == "image-prompt":
                event.input.value = ""
                self._generate_image(text)
                return
            if not text:
                return
            event.input.value = ""
            self._send_chat(text)

        def _send_chat(self, text: str) -> None:
            status = self.query_one("#status", Static)
            status.update("THINKING · 生成中…")
            current = self._transcript_text()
            try:
                if self._session_id:
                    _request_json(
                        self._host,
                        f"/v1/sessions/{self._session_id}/messages",
                        {"owner_scope": "local", "role": "user", "content": text},
                    )
                answer = _stream_chat(
                    self._host,
                    {
                        "model": self._current_model or "harness-default",
                        "messages": [{"role": "user", "content": text}],
                    },
                )
                if self._session_id and answer:
                    _request_json(
                        self._host,
                        f"/v1/sessions/{self._session_id}/messages",
                        {"owner_scope": "local", "role": "assistant", "content": answer},
                    )
                body = answer if answer.strip() else "（模型返回了空内容）"
                self._set_transcript(current + f"\n\nYOU\n{text}\n\nKOAKUMIX\n{body}")
                status.update("ONLINE · 就绪")
            except (RuntimeError, IndexError, AttributeError, TypeError):
                self._set_transcript(current + f"\n\nYOU\n{text}\n\n（后端未连接：已记录输入，未调用模型）")
                status.update("OFFLINE · 后端未连接")

    return HarnessApp()


def _fetch_model_library(host: str, *, timeout: float = 3.0) -> dict[str, Any]:
    """Pull profiles / presets / download jobs.  Pure I/O: never touches widgets.

    超时默认 3s（不是 20s）：这是**本地** harness API，而本函数在启动路径上被调用，
    离线时三个端点各等满会把开窗拖到一分钟。空列表与"连不上"用 ``ok`` 区分。
    """

    library: dict[str, Any] = {}
    ok = False
    try:
        profiles = _request_json(host, "/v1/model-profiles", timeout=timeout)
        library["profiles"] = [item for item in profiles.get("profiles", []) if isinstance(item, dict)]
        ok = True
    except RuntimeError:
        library["profiles"] = []
    try:
        presets = _request_json(host, "/v1/model-presets", timeout=timeout)
        library["presets"] = [item for item in presets.get("presets", []) if isinstance(item, dict)]
        ok = True
    except RuntimeError:
        library["presets"] = []
    try:
        downloads = _request_json(host, "/v1/model-downloads", timeout=timeout)
        library["jobs"] = [item for item in downloads.get("jobs", []) if isinstance(item, dict)]
        ok = True
    except RuntimeError:
        library["jobs"] = []
    library["ok"] = ok
    return library


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app = create_app(
        host=args.host,
        model=args.model,
        llama_exe=args.llama_executable,
        serve=not args.no_serve,
        splash=not args.no_splash,
        splash_min=args.splash_time,
        rag_limit=args.rag_limit,
        rag_mode=args.rag_mode,
        image_dir=args.image_dir,
        image_size=args.image_size,
        image_steps=args.image_steps,
    )
    app.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_IMAGE_STEPS",
    "MAX_SOURCE_BYTES",
    "NAV_ITEMS",
    "QLH_BASE_URL",
    "RAG_INDEX_MODES",
    "build_parser",
    "create_app",
    "format_add_result",
    "format_image_result",
    "format_mcp_result",
    "format_model_library",
    "format_profile_list",
    "format_rag_result",
    "image_output_dir",
    "main",
    "mcp_arguments_template",
    "parse_image_size",
    "parse_mcp_arguments",
    "read_source_file",
    "save_generated_image",
    "start_local_backend",
    "tool_schema",
]
