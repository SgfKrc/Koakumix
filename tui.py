"""Koakumix CLI 工作台 —— Textual TUI。

配色分两层（用户指定）：
- **启动页**保留恐虐红金白黑（见 :mod:`harness_workbench.splash`）；
- **主界面**用中性深色（参照 reasonix cli 的 GitHub Dark 观感：底 #0d1117 / #161b22、
  正文 #c9d1d9、次要 #8b949e、强调紫 #bc8cff 与蓝 #58a6ff）—— 大面积红底黑字长时间
  阅读很吃力。

后端连接：默认**先复用**已有 harness API；若 ``--host`` 无人应答且未禁用，则**自动拉起**
一个内嵌 harness API（复用 :mod:`harness_workbench.desktop` 的装配：QLH 主项目优先，
否则自起 llama-server）。避免一进来就退化成 FIXTURE 离线态。

左栏导航（对话 / 知识库 / 资产 / 运行时）切换主区内容：
- 对话页：新消息自动滚动到底（否则长会话只看到顶部那截，像"没有回应"）；
- 知识库页：**可直接检索**（``POST /v1/rag/search``），展示命中与上下文规模。

启动动画与 ``--no-splash`` / ``--splash-time`` 语义不变；非 TTY 自动跳过。
"""

from __future__ import annotations

import argparse
import json
import threading
import urllib.error
import urllib.request
from typing import Any

DEFAULT_HOST = "http://127.0.0.1:8090"
QLH_BASE_URL = "http://127.0.0.1:8000"

NAV_ITEMS: tuple[tuple[str, str], ...] = (
    ("chat", "对话"),
    ("library", "知识库"),
    ("assets", "资产"),
    ("runtime", "运行时"),
)

RAG_INDEX_MODES: tuple[str, ...] = ("fts", "keyword", "graph")
HIT_SNIPPET_CHARS = 240


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
    return data


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
        lines.append("（没有命中。库为空或关键词不匹配 —— 可先用 POST /v1/rag/sources 添加文档。）")
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
        #rag-query { display: none; border: solid #30363d; background: #0d1117; }
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
            ("3", "nav(2)", "资产"),
            ("4", "nav(3)", "运行时"),
            ("m", "reload_models", "刷新模型"),
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
                        yield Input(placeholder="检索知识库…（回车执行）", id="rag-query")
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

        # ---- panel text ----
        def _set_panel(self, text: str) -> None:
            self.query_one("#panel-text", Static).update(text)
            self._scroll("#panel-scroll")

        # ---- navigation ----
        def action_nav(self, index: int) -> None:
            self._nav = max(0, min(index, len(NAV_ITEMS) - 1))
            self._render_nav()

        def _render_nav(self) -> None:
            """Switch the main area between the chat view and the info panels."""

            chatting = self._nav == 0
            self.query_one("#transcript-scroll").display = chatting
            self.query_one("#composer").display = chatting
            self.query_one("#panel").display = not chatting
            # 检索框只在知识库页出现（其他面板是只读的）。
            self.query_one("#rag-query").display = self._nav == 1
            self._mark_nav()
            if chatting:
                return
            builders = (self._library_panel, self._assets_panel, self._runtime_panel)
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
            if self._last_query:
                return head
            return head + "\n在下方输入框输入关键词并回车即可检索（POST /v1/rag/search）。"

        def _assets_panel(self) -> str:
            image = self._boot.get("image_raw") or {}
            rows = [f"已注册模型 : {len(self._models)}"]
            if self._current_model:
                rows.append(f"当前模型   : {self._current_model}")
            rows.append(f"图像运行时 : {self._boot.get('image', 'unavailable')}")
            if image:
                rows.append(f"  txt2img  : {bool(image.get('supports_txt2img'))}")
                rows.append(f"  runtime  : {bool(image.get('runtime_available'))}")
            return "资产\n\n" + "\n".join(rows) + "\n\n模型列表在左栏 MODEL 区，选中即可切换。"

        def _runtime_panel(self) -> str:
            return (
                "运行时\n\n"
                f"后端地址 : {self._host}\n"
                f"后端类型 : {self._boot.get('backend', 'unavailable')}\n"
                f"内嵌后端 : {'是（本 TUI 拉起）' if self._shell is not None else '否（复用外部服务）'}\n"
                f"会话数   : {len(self._sessions)}\n"
                f"模型数   : {len(self._models)}"
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

        # ---- models ----
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
            if self._nav == 2:
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

        # ---- inputs：检索框与对话框分流 ----
        def on_input_submitted(self, event: Input.Submitted) -> None:
            text = event.value.strip()
            if getattr(event.input, "id", None) == "rag-query":
                event.input.value = ""
                self._run_rag_search(text)
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
    )
    app.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_HOST",
    "NAV_ITEMS",
    "QLH_BASE_URL",
    "RAG_INDEX_MODES",
    "build_parser",
    "create_app",
    "format_rag_result",
    "main",
    "start_local_backend",
]
