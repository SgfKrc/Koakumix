"""Koakumix CLI 工作台 —— 恐虐配色（红白金黑）的 Textual TUI。

配色（Khorne）：黑底 ``#0a0607`` / 血红 ``#8b1a1a`` / 金 ``#e8b923`` / 白 ``#f2ece4``。
启动动画见 :mod:`harness_workbench.splash`：默认开启，``--no-splash`` 关闭，
``--splash-time`` 调最小展示秒数；非 TTY 自动跳过。

后端仍是 harness 的 ``/v1`` 面（``--host`` 默认指向本地 api_layer）。
观感结构参照 Patchouli（``tools/docagent/patchouli/``），主题换成恐虐。

并行约定（沿用 Patchouli 的硬约束：动画不延迟启动）：网络探测在 worker 线程执行，
结果经 ``call_from_thread`` 回主线程更新 widget —— Textual 不允许跨线程触碰 widget。
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Koakumix CLI 工作台（恐虐配色）")
    parser.add_argument("--host", default="http://127.0.0.1:8090", help="harness API base URL")
    parser.add_argument("--model", default="harness-default")
    parser.add_argument("--no-splash", action="store_true", help="跳过启动动画")
    parser.add_argument("--splash-time", type=float, default=1.0, help="启动动画最小展示秒数（默认 1.0；加载更慢时不额外等待）")
    return parser


def _request_json(host: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(
        host.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=8.0) as response:
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
        with urllib.request.urlopen(request, timeout=60.0) as response:
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


def _probe_boot(host: str) -> dict[str, Any]:
    """Collect everything the first screen needs.  Pure I/O: never touches widgets."""

    data: dict[str, Any] = {"online": False}
    try:
        health = _request_json(host, "/healthz")
        data["online"] = True
        data["backend"] = str(health.get("backend", "harness api"))
    except RuntimeError:
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
    return data


def create_app(*, host: str, model: str, splash: bool = True, splash_min: float = 1.0) -> Any:
    try:
        from textual.app import App, ComposeResult
        from textual.containers import Horizontal, Vertical
        from textual.widgets import Button, Footer, Header, Input, Label, ListItem, ListView, Static

        from .splash import SplashScreen
    except ImportError as exc:  # pragma: no cover - optional UI dependency
        raise RuntimeError("Textual is required for the TUI; install the optional harness UI dependency") from exc

    class HarnessApp(App[None]):
        TITLE = "KOAKUMIX"
        SUB_TITLE = "血祭血神，颅献颅座"
        CSS = """
        /* 恐虐配色：黑底 / 血红 / 金 / 白 */
        Screen { background: #0a0607; color: #f2ece4; }
        Header { background: #140a0c; color: #e8b923; }
        Footer { background: #140a0c; color: #9a7b6a; }
        #layout { height: 1fr; }
        #rail { width: 30; padding: 1 2; background: #140a0c; border: solid #8b1a1a; }
        #main { width: 1fr; padding: 1 3; }
        #status { height: 3; color: #e8b923; border-bottom: solid #8b1a1a; }
        #banner { color: #8b1a1a; height: auto; padding: 0 0 1 0; }
        #transcript { height: 1fr; padding: 1 0; overflow-y: auto; }
        #composer { dock: bottom; height: 5; border: solid #8b1a1a; background: #1a0d0f; }
        ListItem { padding: 1; color: #9a7b6a; }
        ListItem:hover { background: #3a1113; color: #e8b923; }
        ListItem:focus { background: #3a1113; color: #f2ece4; }
        Input { background: #1a0d0f; color: #f2ece4; }
        Input:focus { border: solid #e8b923; }
        Button { background: #8b1a1a; color: #f2ece4; border: none; }
        Button:focus { background: #b02323; border: solid #e8b923; }
        .label { color: #e8b923; text-style: bold; }
        .message { padding: 1 0; }
        .muted { color: #8a6f66; }
        .you { color: #f2ece4; }
        .kumix { color: #e8b923; }
        """

        def __init__(self) -> None:
            super().__init__()
            self._splash = bool(splash)
            self._splash_min = float(splash_min)
            self._session_id: str | None = None
            self._sessions: list[dict[str, Any]] = []

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
                    yield Label("SESSION", classes="label")
                    yield Button("+ 新建会话", id="new-session")
                    yield ListView(id="session-list")
                    yield Label("RAG / ASSETS", classes="label")
                    yield Static("等待能力探测", id="utility-status", classes="muted")
                with Vertical(id="main"):
                    yield Static("CHECKING · harness API", id="status")
                    yield Static("血祭血神，颅献颅座。", id="banner")
                    yield Static("KOAKUMIX\n\n等待一条消息。", id="transcript")
                    yield Input(placeholder="输入消息，回车发送…", id="composer")
            yield Footer()

        def on_mount(self) -> None:
            use_splash = self._splash and not self.is_headless
            if use_splash:
                self.push_screen(SplashScreen("点燃血神之炉……", min_show=self._splash_min))
                self.run_worker(self._boot_worker, thread=True, name="boot")
            else:
                self._apply_boot(_probe_boot(host))

        # ---- boot（网络在 worker，UI 更新回主线程）----
        def _boot_worker(self) -> None:
            data = _probe_boot(host)
            self.call_from_thread(self._apply_boot, data)

        def _apply_boot(self, data: dict[str, Any]) -> None:
            status = self.query_one("#status", Static)
            utility = self.query_one("#utility-status", Static)
            if data.get("online"):
                status.update(f"ONLINE · {data.get('backend', 'harness api')}")
                utility.update(f"{data.get('rag', 'RAG unknown')}\n{data.get('image', 'TXT2IMG unknown')}")
                self._sessions = list(data.get("sessions") or [])
                self._render_sessions()
                if self._sessions:
                    self._select_session(str(self._sessions[0].get("session_id", "")))
            else:
                status.update("FIXTURE · API 未连接，发送消息只显示离线提示")
                utility.update("RAG / ASSETS · API unavailable")
            self._close_splash()

        def _close_splash(self) -> None:
            screen = self.screen
            if isinstance(screen, SplashScreen):
                try:
                    screen.notify_loaded()
                except Exception:  # noqa: BLE001 - screen already gone
                    pass

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
                payload = _request_json(host, f"/v1/sessions/{session_id}?owner_scope=local")
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
            if getattr(event.list_view, "id", None) == "session-list":
                self._select_session(str(getattr(event.item, "name", "")))

        def on_button_pressed(self, event: Any) -> None:
            if getattr(event.button, "id", None) != "new-session":
                return
            try:
                created = _request_json(host, "/v1/sessions", {"owner_scope": "local", "title": "New session"})
            except RuntimeError:
                self.query_one("#status", Static).update("FIXTURE · API 未连接，无法新建会话")
                return
            self._session_id = str(created.get("session_id", ""))
            try:
                payload = _request_json(host, "/v1/sessions?owner_scope=local&limit=50")
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
            current = str(transcript.renderable)
            try:
                if self._session_id:
                    _request_json(
                        host,
                        f"/v1/sessions/{self._session_id}/messages",
                        {"owner_scope": "local", "role": "user", "content": text},
                    )
                answer = _stream_chat(host, {"model": model, "messages": [{"role": "user", "content": text}]})
                if self._session_id and answer:
                    _request_json(
                        host,
                        f"/v1/sessions/{self._session_id}/messages",
                        {"owner_scope": "local", "role": "assistant", "content": answer},
                    )
                transcript.update(current + f"\n\nYOU\n{text}\n\nKOAKUMIX\n{answer}")
            except (RuntimeError, IndexError, AttributeError, TypeError):
                transcript.update(current + f"\n\nYOU\n{text}\n\nFIXTURE\nAPI 未连接，已记录输入但未调用模型。")

    return HarnessApp()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app = create_app(
        host=args.host,
        model=args.model,
        splash=not args.no_splash,
        splash_min=args.splash_time,
    )
    app.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "create_app", "main"]
