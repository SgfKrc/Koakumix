"""Koakumix desktop shell: serve the local API and open it in a native WebView window.

Design notes:

* **pywebview** instead of a bundled Chromium: the shell starts the existing
  ``api_layer`` FastAPI app locally and points a system WebView (WebView2 on
  Windows) at it, so the download stays small and the process model stays Python.
* The shell is **opt-in**: ``pywebview`` is declared in the ``desktop`` extra and
  never becomes a hard dependency of the harness itself.
* Everything fails closed and says why: a missing ``webview`` install, a missing
  model path, a missing frontend build or a busy port must not silently start a
  blank window.
* The non-GUI parts (asset discovery, static mounting, port selection, URL
  composition, readiness report) are testable without a display, so the shell can
  be verified in CI and on machines where opening a window is undesirable.

Run it with::

    cd ui_react && npm install && npm run build      # build the UI once
    koakumix-desktop --model models/qwen3-4b-gguf/<file>.gguf
    koakumix-desktop --check                         # readiness report, opens nothing
"""

from __future__ import annotations

import argparse
import contextlib
import json
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .api_layer import create_app


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIST = REPO_ROOT / "ui_react" / "dist"
DEFAULT_ICON = REPO_ROOT / "assets" / "Koakumix.png"
DEFAULT_TITLE = "Koakumix"
SHELL_SCHEMA = "qlh.koakumix.desktop_shell.v1"


class DesktopShellError(RuntimeError):
    """Raised when the desktop shell cannot start for a stated reason."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class DesktopShellConfig:
    dist_dir: Path | str | None = None
    icon: Path | str | None = None
    host: str = "127.0.0.1"
    port: int = 0  # 0 -> pick a free port
    title: str = DEFAULT_TITLE
    width: int = 1280
    height: int = 860
    open_window: bool = True
    extra: Mapping[str, Any] = field(default_factory=dict)

    def resolved_dist(self) -> Path:
        return Path(self.dist_dir).expanduser() if self.dist_dir else DEFAULT_DIST

    def resolved_icon(self) -> Path | None:
        path = Path(self.icon).expanduser() if self.icon else DEFAULT_ICON
        return path if path.is_file() else None


def frontend_status(config: DesktopShellConfig) -> dict[str, Any]:
    """Report whether the frontend build is present, without starting anything."""

    dist = config.resolved_dist()
    index = dist / "index.html"
    status: dict[str, Any] = {
        "dist_dir": str(dist),
        "dist_present": dist.is_dir(),
        "index_present": index.is_file(),
        "icon": str(config.resolved_icon()) if config.resolved_icon() else None,
    }
    if not index.is_file():
        status["hint"] = "run `cd ui_react && npm install && npm run build`"
    return status


def require_frontend(config: DesktopShellConfig) -> Path:
    """Return the dist directory or explain exactly what is missing."""

    dist = config.resolved_dist()
    index = dist / "index.html"
    if not index.is_file():
        raise DesktopShellError(
            f"frontend build not found at {index}; run `cd ui_react && npm install && npm run build`",
            code="frontend_not_built",
        )
    return dist


def require_webview() -> Any:
    """Import pywebview lazily and fail closed with an actionable message."""

    try:
        import webview  # noqa: PLC0415 - optional dependency, imported on demand
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DesktopShellError(
            "pywebview is not installed; install the desktop extra: pip install -e \".[desktop]\"",
            code="pywebview_not_installed",
        ) from exc
    return webview


def webview_available() -> bool:
    """Report whether pywebview can be imported, without importing it."""

    import importlib.util

    return importlib.util.find_spec("webview") is not None


def pick_free_port(host: str) -> int:
    """Ask the OS for an unused port on ``host``."""

    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


class DesktopShell:
    """Own the chat adapter, the API server thread and (optionally) the native window."""

    def __init__(self, config: DesktopShellConfig | None = None, *, app: Any | None = None) -> None:
        self.config = config or DesktopShellConfig()
        self._app = app
        self._adapter: Any | None = None
        self._server: Any | None = None
        self._thread: threading.Thread | None = None
        self._port: int | None = None

    # ------------------------------------------------------------------ setup

    def build_app(self) -> Any:
        """Create the FastAPI app (with its chat adapter) and mount the frontend build."""

        app = self._app or create_app(self._build_adapter())
        dist = self.config.resolved_dist()
        if (dist / "index.html").is_file():
            self._mount_static(app, dist)
        return app

    def build_adapter(self) -> Any:
        """Construct (but do not start) the llama-server adapter, as ``cli.py`` does.

        ``config.extra`` may carry ``model`` (**required**), ``executable``,
        ``llama_host`` / ``llama_port``, ``context_size``, ``max_new_tokens``,
        ``mmproj``, ``enable_jinja`` and ``cache_prompt``.  A missing model is
        refused here with an actionable code instead of surfacing a confusing
        failure from inside the API layer.
        """

        options = dict(self.config.extra)
        model = options.get("model")
        if not model:
            raise DesktopShellError(
                "no chat model configured; pass --model <path.gguf> to start the shell",
                code="model_not_configured",
            )
        from .adapters import LlamaServerAdapter, LlamaServerConfig, LlamaServerProcess  # noqa: PLC0415

        server_config = LlamaServerConfig(
            executable=options.get("executable", "llama-server"),
            model=Path(model),
            host=options.get("llama_host", "127.0.0.1"),
            port=int(options.get("llama_port", 8080)),
            context_size=int(options.get("context_size", 4096)),
            max_new_tokens=int(options.get("max_new_tokens", 768)),
            mmproj=options.get("mmproj"),
            enable_jinja=bool(options.get("enable_jinja", True)),
            cache_prompt=bool(options.get("cache_prompt", True)),
        )
        self._adapter = LlamaServerAdapter(server_config, process=LlamaServerProcess(server_config))
        return self._adapter

    def _build_adapter(self) -> Any:
        return self._adapter or self.build_adapter()

    def shutdown_adapter(self) -> None:
        """Close the adapter this shell started, if any; safe to call repeatedly."""

        adapter = self._adapter
        if adapter is not None:
            with contextlib.suppress(Exception):
                adapter.close()
            self._adapter = None

    @staticmethod
    def _mount_static(app: Any, dist: Path) -> None:
        """Serve the SPA build; guarded so a missing optional dep is explained."""

        try:
            from fastapi.staticfiles import StaticFiles  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - fastapi implied by api_layer
            raise DesktopShellError(
                "fastapi.staticfiles is unavailable; install the desktop extra",
                code="static_files_unavailable",
            ) from exc
        # Mounted after the API routes so it only catches paths they did not claim.
        app.mount("/app", StaticFiles(directory=str(dist), html=True), name="koakumix-ui")

    # ---------------------------------------------------------------- server

    @property
    def port(self) -> int | None:
        return self._port

    def url(self, *, path: str = "/app/") -> str:
        if self._port is None:
            raise DesktopShellError("shell is not running yet", code="shell_not_running")
        return f"http://{self.config.host}:{self._port}{path}"

    def start_server(self, *, wait_seconds: float = 15.0) -> str:
        """Start uvicorn in a daemon thread and return the served URL."""

        try:
            import uvicorn  # noqa: PLC0415 - optional dependency, imported on demand
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise DesktopShellError(
                "uvicorn is not installed; install the desktop extra: pip install -e \".[desktop]\"",
                code="uvicorn_not_installed",
            ) from exc

        self._port = self.config.port or pick_free_port(self.config.host)
        app = self.build_app()
        config = uvicorn.Config(
            app,
            host=self.config.host,
            port=self._port,
            log_level=str(self.config.extra.get("log_level", "warning")),
        )
        server = uvicorn.Server(config)
        server.install_signal_handlers = lambda: None  # the GUI owns the main thread
        thread = threading.Thread(target=server.run, name="koakumix-api", daemon=True)
        thread.start()
        self._server, self._thread = server, thread

        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            if getattr(server, "started", False):
                return self.url()
            if not thread.is_alive():  # pragma: no cover - startup failure path
                self.shutdown()
                raise DesktopShellError("api server thread exited during startup", code="server_start_failed")
            time.sleep(0.05)
        self.shutdown()
        raise DesktopShellError("api server did not become ready in time", code="server_start_timeout")

    def shutdown(self, *, timeout: float = 5.0) -> None:
        """Stop the server thread and the adapter; safe when nothing was started."""

        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._server = None
        self._thread = None
        self.shutdown_adapter()

    # ------------------------------------------------------------------- run

    def run(self) -> str:
        """Start the server and, unless disabled, open the native window."""

        if self.config.open_window:
            require_frontend(self.config)
        url = self.start_server()
        if not self.config.open_window:
            return url

        webview = require_webview()
        icon = self.config.resolved_icon()
        webview.create_window(
            title=self.config.title,
            url=url,
            width=self.config.width,
            height=self.config.height,
        )
        start_kwargs: dict[str, Any] = {}
        if icon is not None:
            start_kwargs["icon"] = str(icon)
        try:
            webview.start(**start_kwargs)
        finally:
            self.shutdown()
        return url

    def status(self) -> dict[str, Any]:
        """Diagnostics that never require a display."""

        return {
            "shell_schema": SHELL_SCHEMA,
            "running": self._thread is not None and self._thread.is_alive(),
            "url": self.url() if self._port is not None else None,
            "host": self.config.host,
            "port": self._port,
            "title": self.config.title,
            "window": self.config.open_window,
            "webview_available": webview_available(),
            "adapter_configured": bool(self.config.extra.get("model")) or self._adapter is not None,
            "frontend": frontend_status(self.config),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="koakumix-desktop", description="Koakumix desktop shell (pywebview)")
    parser.add_argument("--model", default=None, help="path to the chat model (GGUF) served by llama-server")
    parser.add_argument("--llama-executable", default=None, help="llama-server executable (default: from PATH)")
    parser.add_argument("--llama-port", type=int, default=8080, help="port for the llama-server sidecar")
    parser.add_argument("--ctx-size", type=int, default=4096, help="llama-server context size")
    parser.add_argument("--port", type=int, default=0, help="port to serve the UI/API on (0 = pick a free one)")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    parser.add_argument("--dist", default=None, help="frontend dist directory (default ui_react/dist)")
    parser.add_argument("--icon", default=None, help="window icon (default assets/Koakumix.png)")
    parser.add_argument("--print-url", action="store_true", help="start without opening a window and print the URL")
    parser.add_argument("--check", action="store_true", help="report readiness and exit; opens nothing")
    args = parser.parse_args(argv)

    extra: dict[str, Any] = {"llama_port": args.llama_port, "context_size": args.ctx_size}
    if args.model:
        extra["model"] = args.model
    if args.llama_executable:
        extra["executable"] = args.llama_executable

    config = DesktopShellConfig(
        dist_dir=args.dist,
        icon=args.icon,
        host=args.host,
        port=args.port,
        open_window=not args.print_url and not args.check,
        extra=extra,
    )
    shell = DesktopShell(config)

    if args.check:
        report = shell.status()
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["frontend"]["index_present"] else 2

    try:
        url = shell.run()
    except DesktopShellError as exc:
        print(f"[koakumix-desktop] {exc.code}: {exc}", file=sys.stderr)
        return 1
    finally:
        shell.shutdown()

    if args.print_url:
        print(url)
        # Keep serving until interrupted so the URL stays valid for manual checks.
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry
    raise SystemExit(main())
