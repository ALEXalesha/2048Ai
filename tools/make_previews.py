"""Кадры экранов для README и списка игр: docs/preview*.png.

    python tools/make_previews.py

Окно создаётся offscreen (Qt рисует в память), кадр снимается grab() - без экрана и без
чужих окон поверх. Данные игрока - во временной папке, она уходит в Корзину после съёмки.
Партии на кадрах играет ИИ с постоянным зерном, личной статистики на них нет.
"""
import os
import sys
import tempfile
import time

HOME = tempfile.mkdtemp(prefix="ai2048-previews-")
os.environ["2048AI_HOME"] = HOME
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from aikit import offscreen  # noqa: E402

offscreen.setup(force=True)

from PySide6.QtWidgets import QApplication  # noqa: E402

import aikit  # noqa: E402
from game import APP_NAME, GAME_INFO, create_widget  # noqa: E402
from game.core.session import Session  # noqa: E402

SIZE = (1280, 720)
TIMEOUT = 180


def pump(app, seconds):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.003)


def wait(app, cond, timeout=TIMEOUT):
    end = time.time() + timeout
    while time.time() < end and not cond():
        app.processEvents()
        time.sleep(0.003)
    return cond()


def shot(app, window, name):
    for b in (window.game.me_board, window.game.ai_board):
        b.finish_animation()
    pump(app, 0.2)
    path = os.path.join(ROOT, "docs", f"{name}.png")
    window.grab().save(path)
    print(path)


def ai_until(app, game, tile, seed):
    """ИИ играет на максимуме, пока не соберёт плитку tile (или партия не кончится)."""
    game.start_mode("ai")
    game.ai = Session(4, seed=seed)
    game.ai_board.set_grid(game.ai.grid)
    game.set_speed("max")
    game._schedule_ai(0)
    wait(app, lambda: game.ai.max_tile() >= tile or game.ai.over())
    game.toggle_pause()
    wait(app, lambda: not game.runner.busy(), 10)


def main():
    app = QApplication.instance() or QApplication([])
    window, ctx = aikit.make_window(GAME_INFO, create_widget, APP_NAME)
    window.resize(*SIZE)
    window.show()
    game = window.game
    game.set_size(4)
    ctx.set_theme("dark")
    ctx.set_lang("ru")

    ai_until(app, game, 2048, seed=7)
    shot(app, window, "preview")                 # 16:9 для списка игр
    shot(app, window, "preview_ai")

    game.start_mode("play")
    s = Session(4, seed=3, undo=True)
    game.me = s
    for d in [3, 2, 3, 2, 1, 2, 3, 2, 3, 2, 3, 2, 1, 2, 3, 2] * 4:
        s.move(d)
    game.me_board.set_grid(s.grid)
    game.hint()
    wait(app, lambda: game.me_board.hint_dir is not None, 20)
    game.refresh()
    shot(app, window, "preview_play")

    game.start_mode("versus")
    for d in [3, 2, 3, 2, 1, 2, 3, 2] * 6:
        if game.me.over():
            break
        game.human_move(d)
        wait(app, lambda: not game.runner.busy() and game.ai.moves == game.me.moves, 20)
    shot(app, window, "preview_versus")

    ctx.set_theme("light")
    ctx.set_lang("en")
    ai_until(app, game, 1024, seed=11)
    shot(app, window, "preview_ai_light_en")
    game.show_page("stats")
    shot(app, window, "preview_stats_light_en")
    ctx.set_theme("dark")
    ctx.set_lang("ru")
    game.show_page("stats")
    shot(app, window, "preview_stats")
    game.show_page("game")
    window.close()
    app.processEvents()
    recycle(HOME)


def recycle(path):
    """Временную папку - в Корзину, а не насовсем."""
    if os.name != "nt" or not os.path.exists(path):
        return
    import subprocess
    ps = ("Add-Type -AssemblyName Microsoft.VisualBasic; "
          "[Microsoft.VisualBasic.FileIO.FileSystem]::DeleteDirectory("
          f"'{path}','OnlyErrorDialogs','SendToRecycleBin')")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=False, timeout=60)


if __name__ == "__main__":
    main()
