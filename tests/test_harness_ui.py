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


def test_submitting_a_message_does_not_crash():
    """回归守卫：主界面敲回车曾直接把 app 打崩 —— on_input_submitted 读了
    Static.renderable，而 Textual 8.x 的 Static 没有该属性。此前只测了启动路径，
    因此漏掉了交互路径。"""

    import asyncio

    from harness_workbench import tui

    captured: list[str] = []

    async def scenario() -> None:
        app = tui.create_app(host="http://127.0.0.1:1", serve=False, splash=False)
        async with app.run_test() as pilot:
            await pilot.pause()
            composer = app.query_one("#composer")
            composer.focus()  # 未聚焦时按 Enter 不会派发成 Input.Submitted
            await pilot.pause()
            composer.value = "帮我写个短小的俳句"
            await pilot.press("enter")
            await pilot.pause()
            captured.append(str(app.query_one("#transcript").content or ""))

    asyncio.run(scenario())
    assert "帮我写个短小的俳句" in captured[0], "输入应写入 transcript，且不得抛异常"


def test_nav_switches_panels_and_back_to_chat():
    """回归守卫：左栏导航此前没有任何处理函数（点击/回车都无反应）。
    现在应能在对话与只读面板之间切换，且回到对话时输入框恢复。"""

    import asyncio

    from harness_workbench import tui

    async def scenario() -> None:
        app = tui.create_app(host="http://127.0.0.1:1", serve=False, splash=False)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one("#transcript-scroll").display is True
            assert app.query_one("#panel").display is False

            app.action_nav(1)  # 知识库
            await pilot.pause()
            assert app.query_one("#transcript-scroll").display is False
            assert app.query_one("#composer").display is False
            assert app.query_one("#panel").display is True
            assert "知识库" in str(app.query_one("#panel-text").content)

            app.action_nav(3)  # 运行时
            await pilot.pause()
            assert "运行时" in str(app.query_one("#panel-text").content)

            app.action_nav(0)  # 回到对话
            await pilot.pause()
            assert app.query_one("#transcript-scroll").display is True
            assert app.query_one("#composer").display is True

    asyncio.run(scenario())


def test_format_rag_result_renders_hits_and_empty():
    """知识库检索结果的渲染（纯函数，便于离线验证）。"""

    from harness_workbench import tui

    hit = {"source_id": "d1", "chunk_id": "c1", "title": "标题", "text": "正文", "score": 0.5}
    text = tui.format_rag_result(
        "q", {"hits": [hit], "context": {"char_count": 9}}, limit=8, backend="b", chunks=3
    )
    assert "标题" in text and "0.500" in text and "d1 / c1" in text and "9 字符" in text

    empty = tui.format_rag_result("q", {"hits": [], "context": {}}, limit=8, backend="b", chunks=0)
    assert "没有命中" in empty
    assert "入库" in empty, "空结果时要给出去哪里入库的线索"


def test_library_panel_can_search():
    """知识库页此前只是只读 Static；现在应带检索框并把查询打到 /v1/rag/search。"""

    import asyncio

    from harness_workbench import tui

    async def scenario() -> None:
        app = tui.create_app(host="http://127.0.0.1:1", serve=False, splash=False)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_nav(1)
            await pilot.pause()
            assert app.query_one("#panel").display is True
            assert app.query_one("#rag-query").display is True, "知识库页应有检索框"

            app.action_nav(2)  # 资产页只读，不应出现检索框
            await pilot.pause()
            assert app.query_one("#rag-query").display is False

            app.action_nav(1)
            await pilot.pause()
            box = app.query_one("#rag-query")
            box.focus()
            await pilot.pause()
            box.value = "缓存"
            await pilot.press("enter")
            await pilot.pause()
            text = str(app.query_one("#panel-text").content)
            assert "查询「缓存」" in text, "回车应触发检索并回显查询"
            assert "检索失败" in text or "命中" in text

    asyncio.run(scenario())


def test_read_source_file_validates_before_ingest(tmp_path):
    """入库前的本地文件校验（纯函数）：缺失 / 空 / 过大都要给出可读原因。"""

    from harness_workbench import tui

    good = tmp_path / "note.md"
    good.write_text("# 标题\n正文", encoding="utf-8")
    title, text = tui.read_source_file(str(good))
    assert title == "note.md"
    assert "标题" in text and "正文" in text

    with pytest.raises(ValueError, match="找不到文件"):
        tui.read_source_file(str(tmp_path / "nope.md"))

    empty = tmp_path / "empty.md"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="文件为空"):
        tui.read_source_file(str(empty))

    big = tmp_path / "big.md"
    big.write_text("x" * (tui.MAX_SOURCE_BYTES + 1), encoding="utf-8")
    with pytest.raises(ValueError, match="文件过大"):
        tui.read_source_file(str(big))


def test_format_add_result_lists_actual_fields():
    """不硬编码响应字段：把实际键列出来，契约变化时界面上就能看见。"""

    from harness_workbench import tui

    text = tui.format_add_result("/p/a.md", "a.md", "abc", {"source_id": "src_1", "chunks": 3})
    assert "a.md" in text and "src_1" in text
    assert "返回字段 : chunks, source_id" in text


def test_library_panel_can_ingest(tmp_path):
    """知识库页应带入库框，回车把本地文件读进来（离线时表现为入库失败，但确实发起了请求）。"""

    import asyncio

    from harness_workbench import tui

    source = tmp_path / "note.md"
    source.write_text("# 标题\n正文", encoding="utf-8")

    async def scenario() -> None:
        app = tui.create_app(host="http://127.0.0.1:1", serve=False, splash=False)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_nav(1)
            await pilot.pause()
            assert app.query_one("#rag-add").display is True, "知识库页应有入库框"

            box = app.query_one("#rag-add")
            box.focus()
            await pilot.pause()
            box.value = str(source)
            await pilot.press("enter")
            await pilot.pause()
            assert "入库" in str(app.query_one("#panel-text").content)

            app.action_nav(0)
            await pilot.pause()
            assert app.query_one("#rag-add").display is False, "对话页不应显示入库框"

    asyncio.run(scenario())


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
