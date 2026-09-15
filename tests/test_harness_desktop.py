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


def test_default_icon_points_at_the_shipped_mascot():
    """The icon asset the user supplied must stay referenced by default."""

    assert DEFAULT_ICON.name == "Koakumix.png"
    assert DEFAULT_ICON.parent.name == "assets"


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


def test_build_adapter_maps_options_without_starting_anything(tmp_path):
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

    # Constructed, never started: no process is launched by this call.
    assert adapter is shell._adapter
    server_config = getattr(adapter, "config", None)
    assert server_config is not None, "adapter should expose its llama-server config"
    assert server_config.port == 8123
    assert server_config.context_size == 2048
    assert pathlib.Path(str(server_config.model)).name == "model.gguf"

    shell.shutdown_adapter()
    assert shell._adapter is None


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
