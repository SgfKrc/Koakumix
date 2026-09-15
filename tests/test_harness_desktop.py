"""Koakumix desktop shell: readiness, fail-closed paths and static mounting.

Nothing here opens a window: the shell is split so that everything except
``webview.start`` is verifiable without a display, which is what makes
``koakumix-desktop`` safe to check in CI and on headless machines.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from harness_workbench.desktop import (
    DEFAULT_ICON,
    DesktopShell,
    DesktopShellConfig,
    DesktopShellError,
    frontend_status,
    main,
    pick_free_port,
    require_frontend,
    require_webview,
    webview_available,
)


def _dist(tmp_path: pathlib.Path, *, index: bool = True) -> pathlib.Path:
    dist = tmp_path / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "assets").mkdir(exist_ok=True)
    (dist / "assets" / "app.js").write_text("console.log('koakumix')\n", encoding="utf-8")
    if index:
        (dist / "index.html").write_text(
            "<!doctype html><html><body><div id=root></div></body></html>\n", encoding="utf-8"
        )
    return dist


# --------------------------------------------------------------------- readiness
def test_frontend_status_reports_a_missing_build_with_a_hint(tmp_path):
    config = DesktopShellConfig(dist_dir=tmp_path / "nope")
    status = frontend_status(config)

    assert status["dist_present"] is False
    assert status["index_present"] is False
    assert "npm run build" in status["hint"]


def test_frontend_status_reports_a_present_build(tmp_path):
    config = DesktopShellConfig(dist_dir=_dist(tmp_path))
    status = frontend_status(config)

    assert status["dist_present"] is True
    assert status["index_present"] is True
    assert "hint" not in status


def test_require_frontend_fails_closed_with_the_build_command(tmp_path):
    with pytest.raises(DesktopShellError) as excinfo:
        require_frontend(DesktopShellConfig(dist_dir=tmp_path / "missing"))

    assert excinfo.value.code == "frontend_not_built"
    assert "npm" in str(excinfo.value)


def test_default_paths_point_inside_this_repository():
    """Guard the mistake a real ``--check`` run caught: a wrong REPO_ROOT pointed both
    the frontend and the icon at the main repository instead of this one."""

    from harness_workbench.desktop import DEFAULT_DIST, REPO_ROOT

    repo = pathlib.Path(__file__).resolve().parents[1]
    assert REPO_ROOT == repo
    assert DEFAULT_DIST == repo / "ui_react" / "dist"
    # The mascot the user supplied must actually be found by the default config.
    assert DEFAULT_ICON == repo / "assets" / "Koakumix.png"
    assert DEFAULT_ICON.is_file()
    assert DesktopShellConfig().resolved_icon() is not None


def test_window_icon_prefers_the_ico_that_windows_requires():
    """A real window launch caught this: pywebview hands the icon to .NET
    ``System.Drawing.Icon``, which raises "not a valid image for Icon" on a PNG.
    The default must therefore be an existing .ico, and an explicit icon must win."""

    from harness_workbench.desktop import DEFAULT_ICON_ICO

    repo = pathlib.Path(__file__).resolve().parents[1]
    assert DEFAULT_ICON_ICO == repo / "assets" / "Koakumix.ico"
    assert DEFAULT_ICON_ICO.is_file(), "ship the .ico next to the PNG; Windows needs it"
    assert DEFAULT_ICON_ICO.suffix == ".ico"
    # Default resolution must pick the .ico, not the PNG.
    assert DesktopShellConfig().resolved_icon() == DEFAULT_ICON_ICO
    # An explicit icon is honoured as given (and a missing one is still not fatal).
    assert DesktopShellConfig(icon=str(DEFAULT_ICON)).resolved_icon() == DEFAULT_ICON
    assert DesktopShellConfig(icon=str(repo / "nope.ico")).resolved_icon() is None


def test_frontend_build_uses_relative_asset_urls():
    """A real window launch showed a blank workbench: the built ``index.html``
    referenced ``/assets/…`` absolutely, which 404s because the shell mounts the dist
    at ``/app``.  Vite must therefore build with ``base: './'``."""

    import re

    index = pathlib.Path(__file__).resolve().parents[1] / "ui_react" / "dist" / "index.html"
    if not index.is_file():
        pytest.skip("frontend build not present")
    html = index.read_text(encoding="utf-8")
    urls = re.findall(r'(?:src|href)="([^"]+)"', html)
    assert urls, "expected at least one asset reference in the built index.html"
    absolute = [url for url in urls if url.startswith("/")]
    assert not absolute, f"asset URLs must be relative to work under /app, got {absolute}"


def test_icon_resolution_is_optional_not_fatal(tmp_path):
    missing = DesktopShellConfig(icon=tmp_path / "nope.png")
    assert missing.resolved_icon() is None  # no icon is a graceful state


def test_webview_is_optional_and_fails_closed_when_absent():
    # pywebview is deliberately not a hard dependency: either it is installed, or
    # the shell reports an actionable code instead of crashing obscurely.
    assert isinstance(webview_available(), bool)
    if webview_available():  # pragma: no cover - only in a desktop env
        pytest.skip("pywebview installed; the missing-dependency path is unreachable")

    with pytest.raises(DesktopShellError) as excinfo:
        require_webview()
    assert excinfo.value.code == "pywebview_not_installed"
    assert ".[desktop]" in str(excinfo.value)


# ------------------------------------------------------------------------ config
def test_pick_free_port_returns_a_usable_port():
    port = pick_free_port("127.0.0.1")
    assert 1 <= port <= 65535


def test_build_adapter_requires_a_model():
    shell = DesktopShell(DesktopShellConfig(extra={}))

    with pytest.raises(DesktopShellError) as excinfo:
        shell.build_adapter()
    assert excinfo.value.code == "model_not_configured"


def test_build_adapter_maps_options_and_starts_the_process(tmp_path, monkeypatch):
    """A real launch caught the bug this guards: the adapter was constructed but never
    started, so the window pointed at an API whose chat backend was not listening
    (no ``llama-server`` process, nothing on 8080)."""

    import harness_workbench.adapters as adapters

    events: list[str] = []

    class _FakeProcess:
        def __init__(self, config):
            self.config = config

    class _FakeAdapter:
        def __init__(self, config, *, process):
            self.config = config
            self.process = process

        def start(self):
            events.append("start")

        def close(self):
            events.append("close")

    monkeypatch.setattr(adapters, "LlamaServerProcess", _FakeProcess)
    monkeypatch.setattr(adapters, "LlamaServerAdapter", _FakeAdapter)

    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    shell = DesktopShell(
        DesktopShellConfig(
            extra={
                "model": str(model),
                "executable": "llama-server",
                "llama_host": "127.0.0.1",
                "llama_port": 8123,
                "context_size": 2048,
                "max_new_tokens": 256,
            }
        )
    )

    adapter = shell.build_adapter()

    assert adapter is shell._adapter
    assert events == ["start"], "the adapter must be started, not merely constructed"
    assert adapter.config.port == 8123
    assert adapter.config.context_size == 2048
    assert pathlib.Path(str(adapter.config.model)).name == "model.gguf"

    shell.shutdown_adapter()
    assert shell._adapter is None
    assert events == ["start", "close"]


def test_shutdown_is_safe_before_start():
    shell = DesktopShell(DesktopShellConfig())
    shell.shutdown()  # must not raise
    assert shell.port is None
    with pytest.raises(DesktopShellError) as excinfo:
        shell.url()
    assert excinfo.value.code == "shell_not_running"


# ------------------------------------------------------------------ static host
def test_built_frontend_is_mounted_and_served(tmp_path):
    """The shell must serve the SPA build at /app/ and leave the API routes alone."""

    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    dist = _dist(tmp_path)
    app = FastAPI()

    @app.get("/v1/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    shell = DesktopShell(DesktopShellConfig(dist_dir=dist), app=app)
    shell.build_app()  # mounts /app on the injected app

    client = TestClient(app)
    assert client.get("/v1/healthz").json() == {"status": "ok"}
    page = client.get("/app/")
    assert page.status_code == 200
    assert "id=root" in page.text
    assert client.get("/app/assets/app.js").status_code == 200


def test_app_without_a_build_skips_mounting(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi import FastAPI

    shell = DesktopShell(DesktopShellConfig(dist_dir=tmp_path / "missing"), app=FastAPI())
    app = shell.build_app()

    assert not any(getattr(route, "path", "") == "/app" for route in app.routes)


# ----------------------------------------------------------------------- status
def test_status_is_renderable_before_and_after_startup(tmp_path):
    shell = DesktopShell(DesktopShellConfig(dist_dir=_dist(tmp_path)), app=object())
    report = shell.status()

    assert report["shell_schema"].startswith("qlh.koakumix.desktop_shell")
    assert report["running"] is False
    assert report["url"] is None
    assert report["window"] is True
    assert report["webview_available"] is webview_available()
    assert report["frontend"]["index_present"] is True


def test_check_mode_reports_without_opening_a_window(tmp_path, capsys):
    """`--check` must be safe anywhere: no window, no server, no model required."""

    dist = _dist(tmp_path)
    exit_code = main(["--check", "--dist", str(dist)])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["frontend"]["index_present"] is True
    assert report["running"] is False
    assert report["adapter_configured"] is False  # no --model was given


def test_check_mode_exits_nonzero_when_the_build_is_missing(tmp_path, capsys):
    exit_code = main(["--check", "--dist", str(tmp_path / "missing")])
    capsys.readouterr()

    assert exit_code == 2


# --------------------------------------------------------- backend & store wiring
def test_default_data_dir_follows_localappdata(monkeypatch, tmp_path):
    """Harness state belongs in the per-user directory, never inside the checkout."""

    from harness_workbench.desktop import APP_DIR_NAME, REPO_ROOT, default_data_dir

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData"))
    assert default_data_dir() == tmp_path / "AppData" / APP_DIR_NAME

    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    assert default_data_dir() == pathlib.Path.home() / ".koakumix"
    assert REPO_ROOT not in default_data_dir().parents


def test_resolved_data_dir_prefers_the_explicit_value(tmp_path):
    explicit = tmp_path / "custom"
    assert DesktopShellConfig(data_dir=explicit).resolved_data_dir() == explicit


def test_probe_qlh_api_reports_unavailable_instead_of_raising():
    """A missing main-project API is a normal outcome -- the shell falls back, quietly."""

    from harness_workbench.desktop import probe_qlh_api

    assert probe_qlh_api("http://127.0.0.1:1", timeout=0.3) is False  # nothing listens
    assert probe_qlh_api("not-a-url", timeout=0.3) is False  # malformed stays quiet too


def test_unknown_backend_is_refused():
    shell = DesktopShell(DesktopShellConfig(backend="nope"))

    with pytest.raises(DesktopShellError) as excinfo:
        shell.build_adapter()

    assert excinfo.value.code == "unknown_backend"


def test_qlh_backend_is_preferred_when_the_api_answers(monkeypatch):
    """The reported bug: the shell always started its own llama-server, so the model
    catalog/preset panels had nothing behind them.  With the QLH API reachable the shell
    must talk to it -- and must not demand --model."""

    from harness_workbench import desktop as desktop_module
    from harness_workbench.adapters import QLHAdapter

    monkeypatch.setattr(desktop_module, "probe_qlh_api", lambda *_args, **_kwargs: True)
    shell = DesktopShell(DesktopShellConfig(backend="qlh", qlh_base_url="http://127.0.0.1:8090"))

    adapter = shell.build_adapter()

    assert isinstance(adapter, QLHAdapter)
    assert shell._adapter_backend == "qlh"
    assert adapter.config.base_url == "http://127.0.0.1:8090"


def test_llama_fallback_still_complains_about_a_missing_model(monkeypatch):
    """Falling back to the bundled server keeps the old, actionable error."""

    from harness_workbench import desktop as desktop_module

    monkeypatch.setattr(desktop_module, "probe_qlh_api", lambda *_args, **_kwargs: False)
    shell = DesktopShell(DesktopShellConfig(backend="qlh"))  # unreachable -> fallback

    with pytest.raises(DesktopShellError) as excinfo:
        shell.build_adapter()

    assert excinfo.value.code == "model_not_configured"
    assert "8090" in str(excinfo.value)


def test_build_dependencies_wires_the_stores_the_ui_asked_for(tmp_path):
    """The exact reported failure ("new session errors, something about a store"):
    api_layer answers 503 whenever a store is None, so every keyword the shell passes to
    create_app has to be a real, working object."""

    import inspect

    from harness_workbench.api_layer import create_app
    from harness_workbench.session import SessionStore

    shell = DesktopShell(DesktopShellConfig(data_dir=tmp_path / "state"))
    deps = shell.build_dependencies()

    accepted = set(inspect.signature(create_app).parameters)
    assert set(deps) <= accepted, f"create_app would reject {set(deps) - accepted}"
    assert all(value is not None for value in deps.values())

    state = tmp_path / "state"
    assert isinstance(deps["session_store"], SessionStore)
    for name in ("sessions", "rag", "memory"):
        assert (state / f"{name}.sqlite3").is_file(), f"{name} sqlite was not created"

    # The call the UI makes when you press "new session".
    created = deps["session_store"].create(owner_scope="local", title="New session")
    assert created.session_id
    listed = deps["session_store"].list(owner_scope="local", limit=5)
    assert created.session_id in {item.session_id for item in listed}
