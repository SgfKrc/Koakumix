from pathlib import Path

import pytest

from harness_workbench.tui import build_parser, create_app


ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = ROOT / "ui_react"


def test_tui_parser_and_optional_app_contract():
    args = build_parser().parse_args(["--host", "http://127.0.0.1:8090", "--model", "QW1.8B"])
    assert args.host.endswith(":8090")
    assert args.model == "QW1.8B"
    pytest.importorskip("textual")
    app = create_app(host=args.host, model=args.model)
    assert app.TITLE == "KOAKUMIX"
    # 副标题与启动页签名同源，避免两处文案漂移。
    from harness_workbench import splash as splash_module

    assert app.SUB_TITLE == splash_module.SIGNATURE


def test_tui_splash_controls_and_khorne_theme():
    """恐虐主题与启动动画开关都属于 CLI 契约。"""

    defaults = build_parser().parse_args([])
    assert defaults.no_splash is False
    assert defaults.splash_time == pytest.approx(1.0)

    off = build_parser().parse_args(["--no-splash", "--splash-time", "0.2"])
    assert off.no_splash is True
    assert off.splash_time == pytest.approx(0.2)

    from harness_workbench import splash

    assert splash.COLOR_TOP == "#0a0607"     # 黑底
    assert splash.COLOR_BOTTOM == "#8b1a1a"  # 血红填充
    assert splash.COLOR_SCAN == "#e8b923"    # 金色扫描线
    assert splash.COLOR_EDGE == "#f2ece4"    # 白色描边


def test_tui_splash_signoff_matches_patchouli():
    """启动文案与 Patchouli 对齐（用户指定），且启动屏确实被接入。"""

    source = (ROOT / "tui.py").read_text(encoding="utf-8")
    assert "少女祈祷中" in source
    assert "SplashScreen(" in source


def test_tui_uses_persistent_session_contract():
    tui = (ROOT / "tui.py").read_text(encoding="utf-8")
    assert "/v1/sessions?owner_scope=local" in tui
    assert "/v1/sessions/{self._session_id}/messages" in tui
    assert '"stream": True' in tui
    assert "new-session" in tui


def test_cli_tui_flag_delegates_to_the_tui(monkeypatch):
    """`koakumix --tui` 等价于 `koakumix-tui`，且不会触发 --model 必填校验。"""

    from harness_workbench import cli

    seen: list[list[str]] = []

    def _fake_tui_main(argv=None):
        seen.append(list(argv or []))
        return 0

    # cli.main 在函数内 `from .tui import main`，所以 patch 模块属性即可生效。
    monkeypatch.setattr("harness_workbench.tui.main", _fake_tui_main)

    assert cli.main(["--tui", "--host", "http://127.0.0.1:8099"]) == 0
    assert seen == [["--host", "http://127.0.0.1:8099"]], "--tui 应被剔除，其余参数透传"


def test_react_ui_is_independent_and_uses_non_green_cyber_accent():
    package = (UI_ROOT / "package.json").read_text(encoding="utf-8")
    styles = (UI_ROOT / "src" / "styles.css").read_text(encoding="utf-8")
    app = (UI_ROOT / "src" / "App.tsx").read_text(encoding="utf-8")
    data = (UI_ROOT / "src" / "data.ts").read_text(encoding="utf-8")
    assert '"react"' in package and '"vite"' in package
    assert "#63e6ff" in styles and "#ff5bd7" in styles
    assert "#c7ff3d" not in styles.lower()
    assert "/v1/chat/completions" in data
    assert "streamChat" in data
    assert "/v1/sessions" in data
    assert "/v1/rag/search" in data
    assert "/v1/images/generations" in data
    assert "/v1/mcp/manifest" in data
    assert "/v1/mcp/call" in data
    assert "citation-context" in app
    assert "MCP 控制面" in app
    assert "mcp-tool-list" in app
    assert "TXT2IMG BLOCKED" in app
    assert "skip-link" in app
    assert "aria-current" in app
    assert "prefers-reduced-motion" in styles
    assert (UI_ROOT / "scripts" / "visual_smoke.mjs").exists()
    assert '"visual:smoke"' in package
    assert "RAG" in app
