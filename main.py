"""2048 и ИИ, который сам научился в неё играть.

    python main.py                         # игра
    python main.py --selftest итог.txt     # самопроверка без экрана
    python main.py --offscreen             # окно без экрана (для проверок)

Самопроверка нужна прежде всего собранному exe: он без консоли, поэтому итог пишется в
файл. Данные игрока - во временной папке, его статистику она не трогает.
"""
import argparse
import os
import sys
import tempfile
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def selftest(out_path):
    lines = []
    code = 0
    try:
        from aikit import offscreen
        offscreen.setup(force=True)
        os.environ["2048AI_HOME"] = tempfile.mkdtemp(prefix="ai2048-selftest-")
        import numpy as np

        from game.core import game2048 as g
        from game.version import VERSION

        lines.append(f"2048Ai {VERSION} самопроверка")
        lines.append(f"собранная версия: {'да' if getattr(sys, 'frozen', False) else 'нет'}")
        started = time.time()
        rng = np.random.default_rng(7)
        boards, scores, alive = g.new_games(64, rng)
        while alive.any():
            moves = rng.integers(0, 4, size=boards.shape[0])
            boards, scores, alive = g.play_step(boards, scores, alive, moves, rng)
        lines.append(f"правила: 64 случайные партии, средний счёт {scores.mean():.0f}, "
                     f"{time.time() - started:.2f} с")

        # Окно без экрана: ИИ играет 4x4 настоящей моделью, пока не наберёт 300 ходов.
        from PySide6.QtWidgets import QApplication

        import aikit
        from game import APP_NAME, GAME_INFO, create_widget
        from game.ai import levels
        app = QApplication.instance() or QApplication([])
        window, ctx = aikit.make_window(GAME_INFO, create_widget, APP_NAME)
        window.show()
        game = window.game
        model = levels.find_model(levels.MAIN_MODEL)
        lines.append(f"модель: {'есть' if model else 'НЕТ'} ({os.path.basename(model) if model else '-'})")
        game.set_size(4)
        game.start_mode("ai")
        game.set_speed("max")
        started = time.time()
        while game.ai.moves < 300 and not game.ai.over() and time.time() - started < 120:
            app.processEvents()
            time.sleep(0.002)
        image = window.grab().toImage()
        if image.isNull() or image.width() < 100:
            raise RuntimeError("окно не нарисовалось")
        lines.append(f"ИИ: {game.ai.moves} ходов, счёт {game.ai.score}, плитка {game.ai.max_tile()}, "
                     f"кадр {image.width()}x{image.height()}, {time.time() - started:.1f} с")
        if game.ai.moves < 50:
            raise RuntimeError("ИИ почти не ходил")
        t0 = time.time()
        window.close()
        app.processEvents()
        lines.append(f"закрытие: {time.time() - t0:.2f} с, поток ИИ остановлен: "
                     f"{'да' if game.runner.thread.isFinished() else 'нет'}")
        lines.append("ИТОГ: работает")
    except Exception:  # noqa: BLE001 - любая ошибка должна попасть в файл
        lines.append("ИТОГ: ошибка")
        lines.append(traceback.format_exc())
        code = 1
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return code


def main(argv=None):
    ap = argparse.ArgumentParser(description="2048 и ИИ")
    ap.add_argument("--selftest", metavar="ФАЙЛ", nargs="?", const="selftest.txt",
                    help="без окна проверить правила и ИИ и записать итог в файл")
    ap.add_argument("--offscreen", action="store_true", help="окно без экрана")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest(args.selftest)
    if args.offscreen:
        from aikit import offscreen
        offscreen.setup(force=True)
    import aikit
    from PySide6.QtWidgets import QApplication

    from game import APP_NAME, GAME_INFO, create_widget
    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841 - иконке нужно приложение
    from game.ui.icon import icon
    return aikit.run_standalone(GAME_INFO, create_widget, APP_NAME, icon=icon())


if __name__ == "__main__":
    sys.exit(main())
