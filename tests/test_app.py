"""Окно игры - только offscreen: без видимых окон и без захвата мыши.

ИИ в большинстве тестов поддельный и мгновенный (FakeBrain: только очки хода), чтобы
проверять окно, а не модель. Все ожидания - с тайм-аутом.
"""
import os
import subprocess
import sys
import time

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest

import aikit
from aikit.i18n import Translator
from aikit.widgets import Combo
from game import APP_NAME, GAME_INFO, create_widget
from game.ai import expectimax, ntuple
from game.ai.ntuple import NTupleNet
from game.core import game2048 as g
from game.ui import board as board_mod
from game.ui.ai_runner import corner_move, next_tile_chance
from game.ui.texts import TABLES
from game.ui.widget import GameWidget

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeBrain:
    """Мгновенный ИИ: оценка хода - его очки (сеть из нулей, без поиска)."""

    def __init__(self, available=True):
        self.net = NTupleNet(ntuple.TUPLES_4)
        self._available = available
        self.calls = 0

    def available(self, level="strong"):
        return self._available

    def values(self, board, level="strong", depth=None, time_limit=None):
        self.calls += 1
        return expectimax.action_values(self.net, np.array([board], dtype=np.uint64), 1)[0], 1


def wait_until(app, cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.003)
    app.processEvents()
    return cond()


@pytest.fixture
def ctx(qapp, tmp_path):
    c = aikit.Context(str(tmp_path / "data"))
    yield c


@pytest.fixture
def widget(qapp, ctx):
    w = GameWidget(ctx, brain=FakeBrain())
    w.resize(1180, 760)
    w.show()
    qapp.processEvents()
    yield w
    w.shutdown()
    w.close()
    w.deleteLater()
    qapp.processEvents()


def finish(qapp, w):
    for b in (w.me_board, w.ai_board):
        b.finish_animation()
    qapp.processEvents()


# --- контракт --------------------------------------------------------------------------

def test_game_info_follows_the_contract():
    assert {"id", "title", "summary", "preview", "version"} <= set(GAME_INFO)
    assert GAME_INFO["id"].isidentifier()
    assert set(GAME_INFO["title"]) == {"ru", "en"} and set(GAME_INFO["summary"]) == {"ru", "en"}


def test_standalone_window_opens_and_closes_fast(qapp):
    window, ctx = aikit.make_window(GAME_INFO, create_widget, APP_NAME)
    window.show()
    qapp.processEvents()
    assert window.windowTitle().startswith("2048")
    assert window.isVisible()
    # окно offscreen: на экран Алексея ничего не выходит
    assert qapp.platformName() == "offscreen"
    started = time.time()
    window.close()
    qapp.processEvents()
    assert time.time() - started < 2.0
    assert window.game.runner.thread.isFinished()


def test_shutdown_stops_the_ai_thread_within_two_seconds(widget, qapp):
    widget.start_mode("ai")
    widget.set_speed("max")
    assert wait_until(qapp, lambda: widget.ai.moves > 3)
    started = time.time()
    assert widget.shutdown()
    assert time.time() - started < 2.0
    assert widget.runner.thread.isFinished()
    assert not widget.ai_timer.isActive()


def test_translations_have_the_same_keys():
    assert Translator(TABLES).missing() == {}


# --- «Играю сам» ------------------------------------------------------------------------

def test_arrow_keys_move_tiles(widget, qapp):
    widget.start_mode("play")
    cases = [(Qt.Key.Key_Left, (0, 1), (0, 0)), (Qt.Key.Key_Right, (0, 1), (0, 3)),
             (Qt.Key.Key_Down, (0, 3), (3, 3)), (Qt.Key.Key_Up, (3, 3), (0, 3)),
             (Qt.Key.Key_A, (2, 2), (2, 0)), (Qt.Key.Key_S, (1, 0), (3, 0)),
             (Qt.Key.Key_D, (1, 1), (1, 3)), (Qt.Key.Key_W, (2, 1), (0, 1))]
    for key, start, want in cases:
        widget.me.game.grid = [[0] * 4 for _ in range(4)]
        widget.me.game.grid[start[0]][start[1]] = 128
        moves = widget.me.moves
        QTest.keyClick(widget, key)
        finish(qapp, widget)
        assert widget.me.moves == moves + 1, key
        assert widget.me.grid[want[0]][want[1]] == 128, (key, widget.me.grid)
    assert widget.score_vals["score"].text() == str(widget.me.score)


def test_swipe_with_the_mouse(widget, qapp):
    assert board_mod.swipe_direction(60, 5) == 1
    assert board_mod.swipe_direction(-60, 5) == 3
    assert board_mod.swipe_direction(3, -80) == 0
    assert board_mod.swipe_direction(3, 80) == 2
    assert board_mod.swipe_direction(5, 5) is None
    widget.start_mode("play")
    b = widget.me_board
    got = []
    b.swiped.connect(got.append)
    c = b.board_rect().center()
    press = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(c), QPointF(c), Qt.MouseButton.LeftButton,
                        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    end = QPointF(c.x() - 100, c.y() + 10)
    release = QMouseEvent(QMouseEvent.Type.MouseButtonRelease, end, end, Qt.MouseButton.LeftButton,
                          Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    b.mousePressEvent(press)
    b.mouseReleaseEvent(release)
    assert got == [3]


def test_undo_returns_the_previous_position(widget, qapp):
    widget.start_mode("play")
    start = [row[:] for row in widget.me.grid]
    assert not widget.btn_undo.isEnabled()
    assert any(widget.human_move(d) for d in range(4))
    assert widget.btn_undo.isEnabled()
    widget.btn_undo.click()
    assert widget.me.grid == start and widget.me.score == 0
    assert widget.me_board.grid == start


def test_undo_exists_only_when_playing_yourself(widget):
    for mode in ("ai", "versus"):
        widget.start_mode(mode)
        assert not widget.btn_undo.isVisible()
        assert widget.undo() is False


def test_hint_shows_an_arrow_and_all_four_values(widget, qapp):
    widget.start_mode("play")
    widget.me.game.grid = [[2, 2, 0, 0], [0] * 4, [0] * 4, [0] * 4]
    widget.me_board.set_grid(widget.me.grid)
    widget.hint()
    assert wait_until(qapp, lambda: widget.me_board.hint_dir is not None)
    assert widget.me_board.hint_dir in (g.LEFT, g.RIGHT)
    assert len(widget.compass.values) == 4
    assert "Влево" in widget.values_label.text() or "Left" in widget.values_label.text()


def test_game_over_is_recorded_once(widget, qapp):
    widget.start_mode("play")
    # Единственный ход - влево: после него одна пустая клетка, и никакая новая плитка
    # не даёт хода.
    widget.me.game.grid = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 8], [4, 2, 16, 16]]
    assert widget.human_move(g.LEFT)
    assert widget.me.over()
    assert widget.records.get("me", 4)["games"] == 1
    assert widget.records.get("me", 4)["best_tile"] == 32
    assert widget.human_move(g.UP) is False
    widget.new_game()
    assert widget.records.get("me", 4)["games"] == 1
    assert widget.me_board.caption == ""


# --- «Играет ИИ» ------------------------------------------------------------------------

def test_ai_plays_to_the_end_and_is_recorded(widget, qapp):
    widget.start_mode("ai")
    widget.set_speed("max")
    assert wait_until(qapp, lambda: widget.ai.over(), timeout=60)
    assert widget.ai.moves > 30
    assert widget.records.get("ai", 4)["games"] == 1
    assert "ИИ закончил" in widget.status.text() or "AI finished" in widget.status.text()


def test_pause_stops_and_step_makes_exactly_one_move(widget, qapp):
    widget.start_mode("ai")
    widget.set_speed("fast")
    assert wait_until(qapp, lambda: widget.ai.moves >= 2)
    widget.toggle_pause()
    wait_until(qapp, lambda: not widget.runner.busy())
    finish(qapp, widget)
    n = widget.ai.moves
    QTest.qWait(300)
    assert widget.ai.moves == n
    assert widget.btn_step.isEnabled()
    widget.step()
    assert wait_until(qapp, lambda: widget.ai.moves == n + 1)
    QTest.qWait(200)
    assert widget.ai.moves == n + 1
    widget.toggle_pause()
    assert wait_until(qapp, lambda: widget.ai.moves > n + 1)


def test_speed_changes_the_pace(widget, qapp):
    widget.start_mode("ai")
    widget.set_speed("slow")
    QTest.qWait(1200)
    slow = widget.ai.moves
    widget.new_game()
    widget.set_speed("max")
    QTest.qWait(1200)
    assert widget.ai.moves > 3 * max(slow, 1)


# --- «Соревнование» -------------------------------------------------------------------

def test_versus_uses_one_seed_and_ai_answers_each_move(widget, qapp):
    widget.start_mode("versus")
    assert widget.me.seed == widget.ai.seed
    assert widget.me.grid == widget.ai.grid
    assert widget.me_col.isVisible() and widget.ai_col.isVisible()
    for _ in range(5):
        d = next(d for d in range(4) if widget.me.game.legal(d))
        widget.human_move(d)
        assert wait_until(qapp, lambda: widget.ai.moves == widget.me.moves)
        finish(qapp, widget)


def test_versus_result_goes_to_statistics(widget, qapp):
    widget.start_mode("versus")
    stuck = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
    widget.me.game.grid = [row[:] for row in stuck]
    widget.me.game.score = 10
    widget._game_over_me()
    widget._check_versus()
    assert wait_until(qapp, lambda: widget.versus_done, timeout=60)
    e = ctx_entry = widget.ctx.stats.entry(f"versus/4/{widget.level}")
    assert e["games"] == 1 and (e["wins"] + e["losses"] + e["draws"]) == 1
    assert ctx_entry["losses"] == 1             # ИИ доиграл свою партию дальше 10 очков


# --- оформление -----------------------------------------------------------------------

def test_theme_switch_recolors_the_board(widget, qapp, ctx):
    def pixel():
        img = widget.me_board.grab().toImage()
        return img.pixelColor(img.width() // 2, 3).name()
    widget.start_mode("play")
    ctx.set_theme("dark")
    qapp.processEvents()
    dark = pixel()
    ctx.set_theme("light")
    qapp.processEvents()
    assert pixel() != dark


def test_language_switch_retranslates(widget, ctx, qapp):
    ctx.set_lang("en")
    qapp.processEvents()
    assert widget.mode_buttons["play"].text() == "I play"
    assert widget.btn_new.text() == "New game"
    ctx.set_lang("ru")
    qapp.processEvents()
    assert widget.mode_buttons["play"].text() == "Играю сам"


def test_other_board_sizes_say_honestly_ai_is_not_trained(widget, qapp):
    widget.set_size(5)
    assert widget.me.size == 5
    assert "4×4" in widget.note.text()
    assert not widget.level_combo.isEnabled()
    widget.set_size(3)
    widget.start_mode("ai")
    widget.set_speed("max")
    assert wait_until(qapp, lambda: widget.ai.over(), timeout=30)
    # «большие в угол» на других полях в рекорды ИИ не идут: это не ИИ
    assert widget.records.get("ai", 3)["games"] == 0
    widget.set_size(4)


def test_corner_strategy_works_on_any_size():
    assert corner_move([[0, 2, 0], [0, 0, 0], [0, 0, 0]]) == 2           # вниз
    assert corner_move([[2, 4], [4, 2]]) is None


def test_next_tile_chance_from_the_passport():
    passport = {"results": {"strong": {"max_tile_counts": {"1024": 10, "2048": 30, "4096": 60}}}}
    assert next_tile_chance(passport, 2048) == (4096, pytest.approx(60 / 90))
    assert next_tile_chance(passport, 4096) == (8192, 0.0)
    assert next_tile_chance({}, 2048) == (4096, None)


def test_combo_box_draws_its_own_arrow(widget):
    assert isinstance(widget.level_combo, Combo)
    from aikit import theme
    assert "::drop-down" in theme.qss("dark") and "::down-arrow" in theme.qss("dark")


def test_wrapped_labels_never_have_a_fixed_height(widget, qapp):
    from PySide6.QtWidgets import QLabel
    for mode in ("play", "ai", "versus"):
        widget.start_mode(mode)
        for page in ("game", "stats", "settings"):
            widget.show_page(page)
            qapp.processEvents()
            for lab in widget.findChildren(QLabel):
                if lab.wordWrap():
                    assert lab.maximumHeight() == 16777215, lab.text()
    widget.show_page("game")


def test_wrapped_labels_fit_in_the_smallest_window_with_bigger_fonts(qapp, ctx):
    from PySide6.QtWidgets import QLabel
    window, c = aikit.make_window(GAME_INFO, lambda cx: GameWidget(cx, brain=FakeBrain()), APP_NAME)
    window.setStyleSheet(window.styleSheet() + "\n* { font-size: 15px; }")
    window.resize(window.minimumSize())
    window.show()
    for mode in ("play", "ai", "versus"):
        window.game.start_mode(mode)
        qapp.processEvents()
        for lab in window.game.findChildren(QLabel):
            if lab.wordWrap() and lab.isVisible() and lab.text():
                assert lab.heightForWidth(lab.width()) <= lab.height() + 1, (mode, lab.text())
    window.close()


def test_animation_runs_and_finishes(widget, qapp):
    widget.start_mode("play")
    b = widget.me_board
    done = []
    b.animation_finished.connect(lambda: done.append(1))
    d = next(d for d in range(4) if widget.me.game.legal(d))
    widget.human_move(d)
    assert b.animating()
    assert wait_until(qapp, lambda: not b.animating(), timeout=3)
    assert done


# --- самопроверка exe -------------------------------------------------------------------

def test_selftest_runs_in_a_separate_process(tmp_path):
    out = tmp_path / "selftest.txt"
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "main.py"), "--selftest", str(out)],
                            cwd=ROOT, env=env)
    deadline = time.time() + 240
    while proc.poll() is None:              # жив ли процесс - и не дольше тайм-аута
        if time.time() > deadline:
            proc.kill()
            pytest.fail("самопроверка зависла")
        time.sleep(0.2)
    text = out.read_text(encoding="utf-8")
    assert proc.returncode == 0, text
    assert "ИТОГ: работает" in text
