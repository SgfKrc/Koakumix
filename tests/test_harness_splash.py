"""Koakumix 启动动画（splash）与恐虐 TUI 的行为测试。

这里只做无需 TTY 的验证：像素字形/渐显/配色的纯函数行为，以及用 Textual 的
``run_test`` 驾驶真实 App ——启动屏是否被推入、``--no-splash`` 是否跳过。
"""

from __future__ import annotations

import asyncio

import pytest

from harness_workbench import splash as splash_module
from harness_workbench.splash import (
    COLOR_BOTTOM,
    COLOR_EDGE,
    COLOR_SCAN,
    COLOR_TOP,
    GRID,
    GRID_ROWS,
    GRID_W,
    SIGNATURE,
    SplashScreen,
    UPPER,
    WORD,
    render_logo_markup,
    splash_delay,
)


# ------------------------------------------------------------------ splash 纯函数
def test_every_letter_of_the_word_has_a_glyph():
    """A missing glyph would KeyError at import time; keep the font complete."""

    missing = sorted(set(WORD) - set(splash_module.GLYPHS))
    assert not missing, f"missing pixel glyphs: {missing}"
    for letter, rows in splash_module.GLYPHS.items():
        assert len(rows) == GRID_ROWS, f"{letter} must be {GRID_ROWS} rows"
        assert all(len(row) == 3 for row in rows), f"{letter} rows must be 3 wide"


def test_grid_matches_the_word_layout():
    assert GRID_W == len(WORD) * 4 - 1  # 3 px glyph + 1 px gap
    assert len(GRID) == GRID_ROWS


def test_typing_reveals_progressively():
    full = render_logo_markup(GRID, -1, GRID_W)
    early = render_logo_markup(GRID, -1, 4)

    assert early.count(UPPER) < full.count(UPPER)
    assert early.count(UPPER) > 0, "the first columns should already be visible"


def test_scan_line_is_gold_and_bottom_edge_is_white():
    """恐虐配色分工：填充=血红、扫描线=金、下沿描边=白、其余像素上半=黑。"""

    markup = render_logo_markup(GRID, 0, GRID_W)
    assert COLOR_SCAN in markup, "scanned row should use the gold scan colour"
    assert COLOR_EDGE in markup, "bottom row should carry the white edge"
    assert COLOR_TOP in markup and COLOR_BOTTOM in markup


def test_splash_delay_only_ever_shortens_the_wait():
    assert splash_delay(0.3, 1.0) == pytest.approx(0.7)
    assert splash_delay(1.5, 1.0) == 0.0, "a slow load must not be delayed further"
    assert splash_delay(0.0, 0.0) == 0.0


def test_signature_is_non_empty():
    # 不绑定具体文案 —— 签名是用户可改的展示文本。
    assert SIGNATURE.strip()
    assert len(SIGNATURE) > 8


# ------------------------------------------------------------------ TUI 冒烟
def test_splash_screen_renders_when_pushed():
    """启动屏本身可用：能被推入、能渲染出面板。"""

    from harness_workbench.tui import create_app

    async def scenario() -> None:
        app = create_app(host="http://127.0.0.1:1", model="fixture", splash=False)
        async with app.run_test() as pilot:
            app.push_screen(SplashScreen("点燃血神之炉……", min_show=0.0))
            await pilot.pause()
            assert isinstance(app.screen, SplashScreen)
            assert app.screen.query_one("#splash-logo") is not None

    asyncio.run(scenario())


def test_headless_runs_skip_the_splash_automatically():
    """非 TTY（管道 / CI）必须自动跳过动画 —— Patchouli 沿用的硬约束。"""

    from harness_workbench.tui import create_app

    async def scenario() -> None:
        app = create_app(host="http://127.0.0.1:1", model="fixture", splash=True, splash_min=0.0)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.is_headless is True
            assert not isinstance(app.screen, SplashScreen), "headless 运行不应推启动屏"

    asyncio.run(scenario())


def test_no_splash_goes_straight_to_the_workbench():
    from harness_workbench.tui import create_app

    async def scenario() -> None:
        app = create_app(host="http://127.0.0.1:1", model="fixture", splash=False)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert not isinstance(app.screen, SplashScreen)
            # 离线时主界面仍要可用，而不是崩掉：状态行与输入框都在。
            assert app.query_one("#status") is not None
            assert app.query_one("#composer") is not None

    asyncio.run(scenario())


def test_parser_exposes_splash_controls():
    from harness_workbench.tui import build_parser

    args = build_parser().parse_args(["--no-splash", "--splash-time", "0.25"])
    assert args.no_splash is True
    assert args.splash_time == pytest.approx(0.25)

    default = build_parser().parse_args([])
    assert default.no_splash is False
    assert default.splash_time == pytest.approx(1.0)
