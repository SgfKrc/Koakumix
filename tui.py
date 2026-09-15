"""Koakumix CLI 工作台 —— Textual TUI。

配色分两层（用户指定）：
- **启动页**保留恐虐红金白黑（见 :mod:`harness_workbench.splash`）；
- **主界面**用中性深色（参照 reasonix cli 的 GitHub Dark 观感：底 #0d1117 / #161b22、
  正文 #c9d1d9、次要 #8b949e、强调紫 #bc8cff 与蓝 #58a6ff）—— 大面积红底黑字长时间
  阅读很吃力。

后端连接：默认**先复用**已有 harness API；若 ``--host`` 无人应答且未禁用，则**自动拉起**
一个内嵌 harness API（复用 :mod:`harness_workbench.desktop` 的装配：QLH 主项目优先，
否则自起 llama-server）。避免一进来就退化成 FIXTURE 离线态。

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Koakumix CLI 工作台")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"harness API base URL（默认 {DEFAULT_HOST}）")
    parser.add_argument("--model", default=None, help="自起后端时使用的 GGUF 模型路径")
    parser.add_argument("--llama-executable", default=None, help="自起后端时的 llama-server 可执行文件")
    parser.add_argument("--no-serve", action="store_true", help="不自动拉起后端（只连 --host）")
    parser.add_argument("--no-splash", action="store_true", help="跳过启动动画")
    parser.add_argument("--splash-time", type=float, default=1.0, help="启动动画最小展示秒数（默认 1.0）")
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
    """Collect the first screen's data.  Pure I/O: never touches widgets."""

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
    except RuntimeError:
        data["rag"] = "RAG unavailable"
    try:
        image = _request_json(host, "/v1/images/capabilities")
        ready = bool(image.get("runtime_available")) and bool(image.get("supports_txt2img"))
        data["image"] = "TXT2IMG ready" if ready else "TXT2IMG blocked"
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


def create_app(
    *,
    host: str = DEFAULT_HOST,
    model: str | None = None,
    llama_exe: str | None = None,
    serve: bool = True,
    splash: bool = True,
    splash_min: float = 1.0,
    qlh_base_url: str = QLH_BASE_URL,
) -> Any:
    try:
        from textual.app import App, ComposeResult
        from textual.containers import Horizontal, Vertical
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
        #transcript { height: 1fr; padding: 1 0; overflow-y: auto; border-top: solid #30363d; }
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
        .you { color: #c9d1d9; }
        .kumix { color: #58a6ff; }
        .muted { color: #6e7681; }
        """

        BINDINGS = [
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

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal(id="layout"):
                with Vertical(id="rail"):
                    yield Label("WORKSPACE", classes="label")
                    yield ListView(
                        ListItem(Label("▸ 对话")),
                        ListItem(Label("  知识库")),
                        ListItem(Label("  资产")),
                        ListItem(Label("  运行时")),
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
                    yield Static("KOAKUMIX\n\n等待一条消息。", id="transcript")
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
            self._close_splash()

        def _close_splash(self) -> None:
            screen = self.screen
            if isinstance(screen, SplashScreen):
                try:
                    screen.notify_loaded()
                except Exception:  # noqa: BLE001 - screen already gone
                    pass

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
            self.query_one("#transcript", Static).update("\n\n".join(lines) or "KOAKUMIX\n\n等待一条消息。")

        def on_list_view_selected(self, event: Any) -> None:
            which = getattr(event.list_view, "id", None)
            name = str(getattr(event.item, "name", ""))
            if which == "session-list":
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
            self.query_one("#transcript", Static).update("KOAKUMIX\n\n等待一条消息。")

        # ---- chat ----
        def on_input_submitted(self, event: Input.Submitted) -> None:
            text = event.value.strip()
            if not text:
                return
            event.input.value = ""
            transcript = self.query_one("#transcript", Static)
            # Textual 8.x 的 Static 没有 .renderable；正文用官方属性 .content 读回。
            current = str(transcript.content or "")
            status = self.query_one("#status", Static)
            status.update("THINKING · 生成中…")
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
                transcript.update(current + f"\n\nYOU\n{text}\n\nKOAKUMIX\n{answer}")
                status.update("ONLINE · 就绪")
            except (RuntimeError, IndexError, AttributeError, TypeError):
                transcript.update(current + f"\n\nYOU\n{text}\n\n（后端未连接：已记录输入，未调用模型）")
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
    )
    app.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["DEFAULT_HOST", "QLH_BASE_URL", "build_parser", "create_app", "main", "start_local_backend"]
