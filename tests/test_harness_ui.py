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
            # 用 startswith 精确匹配页面标题。原先用 in 断言「运行时」，而资产页的
            # 「图像运行时」也含这三个字 —— 页序一变就假通过了。
            assert str(app.query_one("#panel-text").content).startswith("知识库")

            app.action_nav(3)  # 资产（页序：对话 / 知识库 / MCP / 资产 / 模型库 / 运行时）
            await pilot.pause()
            assert str(app.query_one("#panel-text").content).startswith("资产")

            app.action_nav(4)  # 模型库
            await pilot.pause()
            assert str(app.query_one("#panel-text").content).startswith("模型库")

            app.action_nav(5)  # 运行时
            await pilot.pause()
            assert str(app.query_one("#panel-text").content).startswith("运行时")

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


def test_mcp_arguments_template_keeps_required_only():
    """参数骨架只保留必填字段，避免一屏空值要用户先删。"""

    import json

    from harness_workbench import tui

    schema = {
        "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}, "owner_scope": {"type": "string"}},
        "required": ["query"],
    }
    assert json.loads(tui.mcp_arguments_template(schema)) == {"query": ""}

    # 没有 required 时退回全部属性，并按类型给出占位值
    loose = tui.mcp_arguments_template(
        {"properties": {"a": {"type": "integer"}, "b": {"type": "boolean"}, "owner_scope": {"type": "string"}}}
    )
    assert json.loads(loose) == {"a": 0, "b": False, "owner_scope": "local"}

    assert tui.mcp_arguments_template(None) == "{}"


def test_parse_mcp_arguments_rejects_bad_json():
    from harness_workbench import tui

    assert tui.parse_mcp_arguments("") == {}
    assert tui.parse_mcp_arguments('{"a": 1}') == {"a": 1}
    with pytest.raises(ValueError, match="合法 JSON"):
        tui.parse_mcp_arguments("nope")
    with pytest.raises(ValueError, match="JSON 对象"):
        tui.parse_mcp_arguments("[1, 2]")


def test_format_mcp_result_handles_content_and_error():
    from harness_workbench import tui

    ok = tui.format_mcp_result("t", {"result": {"content": [{"type": "text", "text": "完成"}]}})
    assert "t" in ok and "完成" in ok

    err = tui.format_mcp_result("t", {"error": {"code": -32601, "message": "未知工具"}})
    assert "调用失败" in err and "未知工具" in err


def test_mcp_page_lists_tools_and_calls_selected():
    """MCP 页：工具列表仅该页可见；选中后自动填参数骨架；回车发起调用。"""

    import asyncio
    import json

    from harness_workbench import tui

    tools = [
        {
            "name": "rag_search",
            "description": "搜索",
            "input_schema": {"properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
        {"name": "skill_list", "description": "列技能", "input_schema": {"properties": {}}},
    ]

    async def scenario() -> None:
        app = tui.create_app(host="http://127.0.0.1:1", serve=False, splash=False)
        app._mcp_tools = tools  # 注入工具定义，避免为跑测试起真后端
        async with app.run_test() as pilot:
            await pilot.pause()
            app._render_mcp_tools()
            app.action_nav(2)
            await pilot.pause()
            assert app.query_one("#mcp-tool-list").display is True
            assert app.query_one("#mcp-args").display is True
            assert app.query_one("#rag-query").display is False, "知识库专有控件不应出现在 MCP 页"

            app._select_mcp_tool("rag_search")
            await pilot.pause()
            assert "rag_search" in str(app.query_one("#panel-text").content)
            assert json.loads(app.query_one("#mcp-args").value) == {"query": ""}

            # 离线 host：调用会失败，但必须有反馈（证明确实发起了调用）
            app.query_one("#mcp-args").value = '{"query": "缓存"}'
            await pilot.press("enter")
            await pilot.pause()
            assert "rag_search" in str(app.query_one("#panel-text").content)

            app.action_nav(0)
            await pilot.pause()
            assert app.query_one("#mcp-tool-list").display is False

    asyncio.run(scenario())


def test_parse_image_size_accepts_common_forms_and_rejects_junk():
    from harness_workbench import tui

    assert tui.parse_image_size("512x512") == (512, 512)
    assert tui.parse_image_size("768*512") == (768, 512)
    with pytest.raises(ValueError, match="WxH"):
        tui.parse_image_size("512")
    with pytest.raises(ValueError, match="整数"):
        tui.parse_image_size("axb")
    with pytest.raises(ValueError, match="64..2048"):
        tui.parse_image_size("16x16")


def test_save_generated_image_writes_a_real_file(tmp_path):
    """终端显示不了图片，所以要真落盘并报路径 —— 这里验证它确实写了。"""

    import base64

    from harness_workbench import tui

    blob = b"\x89PNG\r\n\x1a\n" + b"z" * 32
    item = {"b64_json": base64.b64encode(blob).decode(), "metadata": {"seed": 11, "mime_type": "image/png"}}
    path = tui.save_generated_image(item, tmp_path, stamp="20260101-000000")

    assert path.parent == tmp_path
    assert path.name == "koakumix-20260101-000000-seed11.png"
    assert path.read_bytes() == blob

    with pytest.raises(ValueError, match="b64_json"):
        tui.save_generated_image({}, tmp_path)


def test_image_output_dir_defaults_under_the_koakumix_data_dir(tmp_path):
    from harness_workbench import tui

    assert tui.image_output_dir(str(tmp_path)) == tmp_path
    assert tui.image_output_dir(None).name == "images"


def test_assets_panel_can_generate_an_image(tmp_path):
    """资产页应带提示词框，并把请求打到 /v1/images/generations。"""

    import asyncio

    from harness_workbench import tui

    async def scenario() -> None:
        app = tui.create_app(host="http://127.0.0.1:1", serve=False, splash=False, image_dir=str(tmp_path))
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_nav(3)
            await pilot.pause()
            assert app.query_one("#image-prompt").display is True, "资产页应有提示词框"

            box = app.query_one("#image-prompt")
            box.focus()
            await pilot.pause()
            box.value = "a red star"
            await pilot.press("enter")
            await pilot.pause()
            assert "图像生成" in str(app.query_one("#panel-text").content)

            app.action_nav(0)
            await pilot.pause()
            assert app.query_one("#image-prompt").display is False, "对话页不应显示提示词框"

    asyncio.run(scenario())


def test_format_profile_list_notes_the_overflow():
    from harness_workbench import tui

    assert tui.format_profile_list([]) == "（没有画像）"
    many = [{"model_id": f"m{i}"} for i in range(4)]
    text = tui.format_profile_list(many, limit=2)
    assert "m0" in text and "m1" in text and "m2" not in text
    assert "另有 2 个" in text


def test_format_model_library_renders_profiles_and_blocked_presets():
    """按实测字段渲染：画像一览、预设详情、阻塞原因、下载队列与失败原因。"""

    from harness_workbench import tui

    profiles = [{"model_id": "qwen-1_8b", "backend": "pytorch", "format": "safetensors"}]
    presets = [
        {
            "id": "p1",
            "display": "P one",
            "kind": "safetensors",
            "default_engine": "pytorch",
            "default_quant": "int4",
            "installable": True,
            "description": "小模型",
        },
        {"id": "p2", "display": "P two", "installable": False, "blocked_reasons": ["缺 GPU"]},
    ]
    jobs = [
        {"job_id": "j1", "status": "done", "progress": 1.0, "preset_id": "p1"},
        {"job_id": "j2", "status": "failed", "preset_id": "p2", "error": "网络超时"},
    ]

    text = tui.format_model_library(profiles, presets, jobs, selected="p1")
    assert "qwen-1_8b  [pytorch/safetensors]" in text
    assert "选中预设 : p1" in text and "P one" in text and "可安装   : True" in text
    assert "done" in text
    assert "网络超时" in text, "失败任务的错误信息要显示出来"

    blocked = tui.format_model_library(profiles, presets, jobs, selected="p2")
    assert "阻塞原因 : 缺 GPU" in blocked


def test_model_library_page_and_refresh_report_truthfully():
    """模型库页：列表仅该页可见；t 刷新必须如实报告连不上。

    回归守卫：早先的 `t` 无条件显示「已刷新」，离线时也在骗人。
    """

    import asyncio

    from harness_workbench import tui

    async def scenario() -> None:
        app = tui.create_app(host="http://127.0.0.1:1", serve=False, splash=False)
        app._profiles = [{"model_id": "qwen-1_8b", "backend": "pytorch", "format": "safetensors"}]
        app._presets = [{"id": "p1", "display": "P one", "installable": True}]
        app._jobs = [{"job_id": "j1", "status": "done", "progress": 1.0, "preset_id": "p1"}]
        async with app.run_test() as pilot:
            await pilot.pause()
            app._render_presets()
            app.action_nav(4)
            await pilot.pause()
            assert app.query_one("#preset-list").display is True, "模型库页应有预设列表"
            assert str(app.query_one("#panel-text").content).startswith("模型库")

            app._select_preset("p1")
            await pilot.pause()
            assert "选中预设 : p1" in str(app.query_one("#panel-text").content)

            app.action_refresh()
            await pilot.pause()
            status = str(app.query_one("#status").content)
            assert status.startswith("OFFLINE"), f"离线时不得谎报成功，实际: {status}"

            app.action_nav(0)
            await pilot.pause()
            assert app.query_one("#preset-list").display is False

    asyncio.run(scenario())


def test_jsonrpc_payload_detection():
    """MCP 参数框靠这个判断走工具调用还是通用 JSON-RPC 通道。"""

    from harness_workbench import tui

    assert tui.is_jsonrpc_payload('{"jsonrpc":"2.0","method":"tools/list","id":"1"}') is True
    assert tui.is_jsonrpc_payload('{"query": "缓存"}') is False, "工具参数不是信封"
    assert tui.is_jsonrpc_payload("not json") is False
    assert tui.is_jsonrpc_payload("") is False


def test_parse_image_input_routes_asset_ids_and_prompts():
    from harness_workbench import tui

    assert tui.parse_image_input("@img_abc") == ("asset", "img_abc")
    assert tui.parse_image_input("  @x  ") == ("asset", "x")
    assert tui.parse_image_input("a red star") == ("prompt", "a red star")
    assert tui.parse_image_input("  ") == ("prompt", "")


def test_format_capabilities_uses_the_observed_keys():
    from harness_workbench import tui

    lines = tui.format_capabilities(
        {"backend": "llama_server", "supports_stream": True, "model_ids": ["a", "b"]}
    )
    joined = "\n".join(lines)
    assert "supports_stream" in joined and "2 项" in joined
    assert "返回字段" in joined
    assert tui.format_capabilities(None) == ["能力面   : unavailable"]


def test_format_mcp_manifest_summarises_the_server():
    from harness_workbench import tui

    manifest = {
        "server": {"protocolVersion": "2024-11-05", "serverInfo": {"name": "koakumix", "version": "0.1"}},
        "external_mcp": {"configurations": [], "configuration_only": True},
        "transports": {"jsonrpc": {}, "call": {}, "stdio": {}},
    }
    joined = "\n".join(tui.format_mcp_manifest(manifest, tool_count=16))
    assert "koakumix 0.1" in joined and "2024-11-05" in joined and "16" in joined
    assert "仅配置，未真连" in joined


def test_probe_fails_fast_with_a_compressed_timeout():
    """启动路径必须能压缩超时，否则离线时 9 个端点各等满会把开窗拖到一分钟。"""

    import time

    from harness_workbench import tui

    started = time.monotonic()
    data = tui._probe("http://127.0.0.1:1", timeout=1.0)
    elapsed = time.monotonic() - started
    assert data.get("error"), "离线时应返回 error 而不是抛异常"
    assert elapsed < 5.0, f"离线探测应快速失败，实际 {elapsed:.2f}s"


def test_mcp_rpc_and_asset_prefix_routing():
    """MCP 参数框写 JSON-RPC 信封走 /rpc；资产框写 @id 走取回。"""

    import asyncio

    from harness_workbench import tui

    async def scenario() -> None:
        app = tui.create_app(host="http://127.0.0.1:1", serve=False, splash=False)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_nav(2)
            await pilot.pause()
            box = app.query_one("#mcp-args")
            box.focus()
            await pilot.pause()
            box.value = '{"jsonrpc":"2.0","id":"t","method":"tools/list","params":{}}'
            await pilot.press("enter")
            await pilot.pause()
            assert "JSON-RPC" in str(app.query_one("#panel-text").content)

            app.action_nav(3)
            await pilot.pause()
            image_box = app.query_one("#image-prompt")
            image_box.focus()
            await pilot.pause()
            image_box.value = "@img_abc"
            await pilot.press("enter")
            await pilot.pause()
            assert "取回资产" in str(app.query_one("#panel-text").content)

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
