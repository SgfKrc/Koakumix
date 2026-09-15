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
import os
import socket
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .api_layer import create_app


# `desktop.py` lives at the package root, which *is* the repository root.
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DIST = REPO_ROOT / "ui_react" / "dist"
# pywebview on Windows hands the icon to .NET ``System.Drawing.Icon``, which accepts
# **only .ico** files — a PNG raises "not a valid image for Icon" at window creation.
# Ship both: prefer the .ico for the window, keep the PNG for other backends and docs.
DEFAULT_ICON = REPO_ROOT / "assets" / "Koakumix.png"
DEFAULT_ICON_ICO = REPO_ROOT / "assets" / "Koakumix.ico"
DEFAULT_TITLE = "Koakumix"
SHELL_SCHEMA = "qlh.koakumix.desktop_shell.v1"
# The QLH main-project API is the only backend that has the model-selection /
# model-load / device-profile surface: api_layer reflects model_* onto the adapter, and
# checks each store.  Port 8000 is the main project's own default -- src/config.py
# declares ``API_PORT = _env_int("QLH_API_PORT", 8000)`` and src/api_server.py documents
# ``uvicorn src.api_server:app --port 8000``.  This is *not* the 8090 that
# ui_react/vite.config.ts proxies to; that one is the harness's own api_layer in dev mode.
# When nothing answers here, the shell falls back to its bundled llama-server.
DEFAULT_QLH_BASE_URL = "http://127.0.0.1:8000"
QLH_HEALTH_PATH = "/api/health"
BACKEND_QLH = "qlh"
BACKEND_LLAMA = "llama"
BACKENDS = (BACKEND_QLH, BACKEND_LLAMA)
APP_DIR_NAME = "Koakumix"


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
    # "qlh" prefers the main-project API (full model/session surface); "llama" always
    # starts the bundled llama-server (self-contained, chat only).
    backend: str = BACKEND_QLH
    qlh_base_url: str = DEFAULT_QLH_BASE_URL
    data_dir: Path | str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def resolved_dist(self) -> Path:
        return Path(self.dist_dir).expanduser() if self.dist_dir else DEFAULT_DIST

    def resolved_data_dir(self) -> Path:
        """Where sessions/memory/RAG/images live (SQLite + blobs).

        Defaults to the Windows convention ``%LOCALAPPDATA%\\Koakumix`` so the harness
        never writes user data into the checkout; falls back to ``~/.koakumix`` where
        that variable does not exist.
        """

        if self.data_dir:
            return Path(self.data_dir).expanduser()
        return default_data_dir()

    def resolved_icon(self) -> Path | None:
        """Return the window icon, preferring the ``.ico`` that Windows/.NET requires.

        An explicitly supplied ``icon`` is honoured as given (use ``.ico`` on Windows);
        the default path prefers ``Koakumix.ico`` and falls back to the PNG so other
        backends — and the documentation — keep working off the same asset.
        """

        if self.icon:
            explicit = Path(self.icon).expanduser()
            return explicit if explicit.is_file() else None
        for candidate in (DEFAULT_ICON_ICO, DEFAULT_ICON):
            if candidate.is_file():
                return candidate
        return None


def default_data_dir() -> Path:
    """Return the per-user directory that holds harness state.

    ``%LOCALAPPDATA%\\Koakumix`` on Windows -- the platform convention, and outside the
    checkout on purpose -- falling back to ``~/.koakumix`` where that is unset.
    """

    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / APP_DIR_NAME
    return Path.home() / ".koakumix"


def probe_qlh_api(base_url: str, *, timeout: float = 1.5) -> bool:
    """Cheaply decide whether a QLH main-project API answers at ``base_url``.

    Only ``/api/health`` is touched, over a deliberately short timeout, and *every*
    failure (refused, timed out, non-2xx, malformed URL) reports "not available" so the
    caller can fall back to the bundled llama-server instead of failing the launch.
    """

    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}{QLH_HEALTH_PATH}", timeout=timeout) as response:  # noqa: S310
            return 200 <= int(response.status) < 300
    except Exception:
        return False


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
        self._adapter_backend: str | None = None
        self._server: Any | None = None
        self._thread: threading.Thread | None = None
        self._port: int | None = None

    # ------------------------------------------------------------------ setup

    def build_app(self) -> Any:
        """Create the FastAPI app, inject the harness stores, and mount the frontend build.

        Injecting the stores is what makes the UI work at all: ``api_layer`` answers 503
        on ``/v1/sessions``, ``/v1/rag/*``, ``/v1/images/*`` and ``/v1/mcp/*`` whenever
        the matching store is ``None`` -- which is exactly what an unwired shell did.
        """

        app = self._app or create_app(self._build_adapter(), **self.build_dependencies())
        dist = self.config.resolved_dist()
        if (dist / "index.html").is_file():
            self._mount_static(app, dist)
        return app

    def build_dependencies(self) -> dict[str, Any]:
        """Assemble the user-owned stores backing the harness tool surface.

        Everything is rooted at ``config.resolved_data_dir()``, so the shell owns only
        the files it was explicitly given.  The image engine is assembled fail-closed:
        without an asset root it is the documented *unavailable* engine, which reports
        why instead of pretending generation works.
        """

        data = self.config.resolved_data_dir()
        data.mkdir(parents=True, exist_ok=True)
        from .image import ImageAssetStore, build_local_image_engine  # noqa: PLC0415
        from .mcp_server import HarnessMCPDependencies, ToolRegistry, create_harness_server  # noqa: PLC0415
        from .memory import MemoryStore  # noqa: PLC0415
        from .rag import RagStore  # noqa: PLC0415
        from .session import SessionStore  # noqa: PLC0415

        session_store = SessionStore(data / "sessions.sqlite3")
        rag_store = RagStore(data / "rag.sqlite3")
        memory_store = MemoryStore(data / "memory.sqlite3")
        image_store = ImageAssetStore(data / "images")
        image_engine = build_local_image_engine()
        mcp_server = create_harness_server(
            dependencies=HarnessMCPDependencies(
                session_store=session_store,
                rag_store=rag_store,
                memory_store=memory_store,
                image_adapter=image_engine.engine,
                image_store=image_store,
            ),
            registry=ToolRegistry(),
        )
        return {
            "session_store": session_store,
            "rag_store": rag_store,
            "memory_store": memory_store,
            "image_store": image_store,
            "image_adapter": image_engine.engine,
            "mcp_server": mcp_server,
        }

    def build_adapter(self) -> Any:
        """Build the chat adapter: prefer the QLH main-project API, else llama-server.

        ``backend="qlh"`` (the default) probes ``config.qlh_base_url``; when the
        main-project API answers, the shell talks to it directly.  That is the only path
        where the model-selection / model-load / device panels have data, because
        ``api_layer`` reflects ``model_catalog`` / ``model_presets`` / ``load_model_asset``
        onto the adapter.  When QLH is not running the shell falls back to a bundled
        llama-server so the workbench still opens (chat only).

        ``backend="llama"`` skips the probe and always starts the bundled server.
        ``config.extra`` may carry ``model`` (required on the llama path), ``executable``,
        ``llama_host`` / ``llama_port``, ``context_size``, ``max_new_tokens``,
        ``mmproj``, ``enable_jinja`` and ``cache_prompt``.

        Starting is part of building: a constructed-but-unstarted llama-server adapter
        leaves the window pointing at an API whose chat backend is not listening -- a
        real launch run caught exactly that (no ``llama-server`` process, nothing on
        8080).
        """

        if self._adapter is not None:
            return self._adapter
        if self.config.backend not in BACKENDS:
            raise DesktopShellError(
                f"unknown backend {self.config.backend!r}; expected one of {BACKENDS}",
                code="unknown_backend",
            )
        options = dict(self.config.extra)
        if self.config.backend == BACKEND_QLH and probe_qlh_api(self.config.qlh_base_url):
            from .adapters import QLHAdapter, QLHAdapterConfig  # noqa: PLC0415

            adapter = QLHAdapter(
                QLHAdapterConfig(
                    base_url=self.config.qlh_base_url,
                    model_id=str(options.get("model_id") or "qlh-default"),
                    routing_preference=str(options.get("routing_preference") or "auto"),
                )
            )
            self._adapter_backend = BACKEND_QLH
            self._adapter = adapter
            return adapter

        model = options.get("model")
        if not model:
            raise DesktopShellError(
                "no chat model configured: start the QLH main-project API at "
                f"{self.config.qlh_base_url}, or pass --model <path.gguf> to run the "
                "bundled llama-server",
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
        adapter = LlamaServerAdapter(server_config, process=LlamaServerProcess(server_config))
        adapter.start()
        self._adapter_backend = BACKEND_LLAMA
        self._adapter = adapter
        return adapter

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
            # Windows: taskbar identity, so the window is grouped/labelled as Koakumix
            # instead of falling back to the host interpreter's icon and name.
            if sys.platform == "win32":
                with contextlib.suppress(Exception):
                    import ctypes  # noqa: PLC0415

                    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Koakumix")
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
            # Must be an .ico on Windows: pywebview hands it to .NET System.Drawing.Icon,
            # which rejects PNGs with "not a valid image for Icon".
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
    parser.add_argument("--model", default=None, help="path to the chat model (GGUF) for the bundled llama-server fallback")
    parser.add_argument("--llama-executable", default=None, help="llama-server executable (default: from PATH)")
    parser.add_argument("--llama-port", type=int, default=8080, help="port for the llama-server sidecar")
    parser.add_argument("--ctx-size", type=int, default=4096, help="llama-server context size")
    parser.add_argument("--port", type=int, default=0, help="port to serve the UI/API on (0 = pick a free one)")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    parser.add_argument("--backend", default=BACKEND_QLH, choices=BACKENDS, help="chat backend: qlh (main-project API, full model surface) or llama (bundled llama-server)")
    parser.add_argument("--qlh-base-url", default=DEFAULT_QLH_BASE_URL, help=f"QLH main-project API base URL (default {DEFAULT_QLH_BASE_URL})")
    parser.add_argument("--data-dir", default=None, help="where sessions/memory/RAG/images live (default %%LOCALAPPDATA%%\\Koakumix)")
    parser.add_argument("--dist", default=None, help="frontend dist directory (default ui_react/dist)")
    parser.add_argument("--icon", default=None, help="window icon (default assets/Koakumix.ico)")
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
        backend=args.backend,
        qlh_base_url=args.qlh_base_url,
        data_dir=args.data_dir,
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
