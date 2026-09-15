"""Koakumix 启动动画（splash v1）——逐格半块像素字 + 扫描线 + 打字机。

视觉设计（恐虐配色：红 #8b1a1a / 金 #e8b923 / 白 #f2ece4 / 黑 #0a0607）：
- 标题 ``KOAKUMIX`` 用 **▀ 半块字符逐格上色**：每个像素格 = 上半黑 / 下半红
  （``▀`` fg=#0a0607 on bg=#8b1a1a）——颜色贴合字形、无底色溢出；
  字形最底行像素用白色（上白下红）作为**下沿白描边**；
- 动画：打字机逐列显现 + **扫描线**（亮金半块自上下扫）；
- 状态行：窄红条（3 行）+ spinner 转圈；任意键或点击跳过。

结构与配色位对齐 Patchouli 的 splash v5（``tools/docagent/patchouli/splash.py``，色孽调），
只替换主题色、字形与签名。

硬约束（沿用 Patchouli 的）：与数据加载**并行**、加载完 + 动画播完才关闭（不延长启动）；
``--no-splash`` 可关；非 TTY 由调用方跳过。
"""

from __future__ import annotations

import time

from textual.app import ComposeResult
from textual.containers import Center, Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

# ---- 5x3 像素字形（# 实心）----
GLYPHS: dict[str, list[str]] = {
    "K": ["#.#", "#.#", "##.", "#.#", "#.#"],
    "O": ["###", "#.#", "#.#", "#.#", "###"],
    "A": ["###", "#.#", "###", "#.#", "#.#"],
    "U": ["#.#", "#.#", "#.#", "#.#", "###"],
    "M": ["#.#", "###", "#.#", "#.#", "#.#"],
    "I": ["###", ".#.", ".#.", ".#.", "###"],
    "X": ["#.#", "#.#", ".#.", "#.#", "#.#"],
}

COLOR_TOP = "#0a0607"     # 像素上半：黑
COLOR_BOTTOM = "#8b1a1a"  # 像素下半：红（恐虐）
COLOR_SCAN = "#e8b923"    # 扫描线：金
COLOR_EDGE = "#f2ece4"    # 下沿描边：白
UPPER = "\u2580"          # ▀ 上半块（fg 占上半、bg 占下半）

WORD = "KOAKUMIX"
GRID_H = 5


def splash_delay(elapsed: float, min_show: float) -> float:
    """需补足的等待秒数：加载快于最小展示时补齐（可感知）；加载更慢则不额外等待。"""

    return max(0.0, float(min_show) - float(elapsed))


def _build_grid(word: str = WORD) -> list[list[str]]:
    """字形网格（F=实心，空=透明）；不含膨胀描边（颜色贴合字形）。"""

    width = len(word) * 4 - 1
    grid = [[" "] * width for _ in range(GRID_H)]
    for i, ch in enumerate(word):
        glyph = GLYPHS[ch]
        for r in range(GRID_H):
            for c in range(3):
                if glyph[r][c] == "#":
                    grid[r][i * 4 + c] = "F"
    return grid


GRID = _build_grid()
GRID_W = len(GRID[0])
GRID_ROWS = len(GRID)
LAST_ROW = GRID_ROWS - 1


def _style_for(row: int, scan_row: int) -> str:
    if row == scan_row:
        return f"{COLOR_SCAN} on {COLOR_BOTTOM}"
    if row == LAST_ROW:
        return f"{COLOR_EDGE} on {COLOR_BOTTOM}"
    return f"{COLOR_TOP} on {COLOR_BOTTOM}"


def render_logo_markup(grid: list[list[str]], scan_row: int = -1, revealed_cols: int = 10**9) -> str:
    """网格 → Textual markup（RLE 合并同色）：打字机（revealed_cols）+ 扫描线（scan_row）。"""

    rows = []
    for r in range(len(grid)):
        style = _style_for(r, scan_row)
        parts: list[str] = []
        run = 0
        for c in range(len(grid[0])):
            if grid[r][c] == "F" and c < revealed_cols:
                run += 1
            else:
                if run:
                    parts.append(f"[{style}]{UPPER * run}[/]")
                    run = 0
                parts.append(" ")
        if run:
            parts.append(f"[{style}]{UPPER * run}[/]")
        rows.append("".join(parts))
    return "\n".join(rows)


FRAMES = ("/", "-", "\\", "|")
TICK_SECONDS = 0.08
TYPING_COLS_PER_TICK = 3.75  # 打字速度
SIGNATURE = "Evangelium vom Himmelsturz."  # 启动签名
SIG_CHARS_PER_TICK = 1  # 签名流式速度（字符/tick，从左往右）
SIG_CURSOR = "\u258c"  # ▌
SCAN_STEP_TICKS = 2  # 每 2 tick 扫描线下移一行
HOLD_TICKS = 10  # 扫描完毕后静止 hold（看清标题）


class SplashScreen(ModalScreen):
    """启动屏：KOAKUMIX 半块像素字（上黑下红、底沿白）+ 金色扫描线 + 窄红条。"""

    CSS = """
    SplashScreen { background: #000000 90%; }
    #splash-box { border: round #8b1a1a; background: #0a0607; padding: 0 3; height: auto; }
    #splash-logo { width: auto; margin: 1 0 0 0; }
    #splash-sign { color: #e8b923; height: 1; margin-top: 1; width: 1fr; text-align: center; }
    #splash-line { background: #8b1a1a; color: #f2ece4; height: 3; margin-top: 1; content-align: center middle; }
    """

    def __init__(self, status: str = "启动中…", min_show: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.status_text = status
        self.min_show = float(min_show)
        self._frame = 0
        self._timer = None
        self._t0: float | None = None
        self._loaded = False
        self._typing_done_frame: int | None = None
        self._sig_done_frame: int | None = None

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="splash-box"):
                yield Static(render_logo_markup(GRID, -1, TYPING_COLS_PER_TICK * 2), id="splash-logo")
                yield Static("", id="splash-sign", markup=False)
                yield Static("", id="splash-line", markup=False)

    def on_mount(self) -> None:
        self._t0 = time.monotonic()
        self._render_line()
        self._timer = self.set_interval(TICK_SECONDS, self._tick)

    def _tick(self) -> None:
        self._frame += 1
        revealed = min(GRID_W, int(TYPING_COLS_PER_TICK * (self._frame + 2)))
        typing_done = revealed >= GRID_W
        if typing_done and self._typing_done_frame is None:
            self._typing_done_frame = self._frame
        sig_revealed = min(len(SIGNATURE), SIG_CHARS_PER_TICK * self._frame)
        if sig_revealed >= len(SIGNATURE) and self._sig_done_frame is None:
            self._sig_done_frame = self._frame
        try:
            sign_text = SIGNATURE[:sig_revealed]
            if sig_revealed < len(SIGNATURE):
                sign_text += SIG_CURSOR
            self.query_one("#splash-sign", Static).update(sign_text)
        except Exception:  # noqa: BLE001 — 屏幕已卸载
            pass
        scan_row = -1
        if typing_done:
            since = self._frame - self._typing_done_frame
            if since < GRID_ROWS * SCAN_STEP_TICKS:  # 扫描一轮后进入静止 hold
                scan_row = (since // SCAN_STEP_TICKS) % GRID_ROWS
        try:
            self.query_one("#splash-logo", Static).update(render_logo_markup(GRID, scan_row, revealed))
        except Exception:  # noqa: BLE001 — 屏幕已卸载
            pass
        self._render_line()
        self._maybe_finish()

    def _anim_done(self) -> bool:
        if self._typing_done_frame is None or self._sig_done_frame is None:
            return False
        title_hold = (self._frame - self._typing_done_frame) >= (GRID_ROWS * SCAN_STEP_TICKS + HOLD_TICKS)
        sig_hold = (self._frame - self._sig_done_frame) >= HOLD_TICKS
        return title_hold and sig_hold

    def notify_loaded(self) -> None:
        """数据加载完成：动画播完（打字+扫描+hold）且 min_show 满足后自行关闭。"""

        self._loaded = True
        self._maybe_finish()

    def _maybe_finish(self) -> None:
        if not (self._loaded and self._anim_done()):
            return
        if self._t0 is not None and (time.monotonic() - self._t0) < self.min_show:
            return
        try:
            self.dismiss()
        except Exception:  # noqa: BLE001
            pass

    def _render_line(self) -> None:
        spin = FRAMES[self._frame % len(FRAMES)]
        try:
            self.query_one("#splash-line", Static).update(f"  [{spin}] {self.status_text}    （任意键跳过）")
        except Exception:  # noqa: BLE001 — 屏幕已卸载
            pass

    def set_status(self, text: str) -> None:
        self.status_text = text
        self._render_line()

    def on_key(self) -> None:
        self.dismiss()

    def on_click(self) -> None:
        self.dismiss()


__all__ = [
    "COLOR_BOTTOM",
    "COLOR_EDGE",
    "COLOR_SCAN",
    "COLOR_TOP",
    "GRID",
    "GRID_H",
    "GRID_ROWS",
    "GRID_W",
    "SIGNATURE",
    "SplashScreen",
    "WORD",
    "render_logo_markup",
    "splash_delay",
]
