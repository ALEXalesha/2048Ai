"""Мутационная проверка: портим код нарочно и убеждаемся, что тесты это ловят.

    python tools/mutation_check.py            # все группы
    python tools/mutation_check.py rules ai   # только эти

Каждая мутация - одна замена строки в одном файле. Файл правится на месте и всегда
возвращается обратно (finally), байт в байт. Тесты идут в отдельном процессе без записи
.pyc (PYTHONDONTWRITEBYTECODE): иначе испорченный байткод мог бы пережить возврат файла.
Выживший мутант - дыра в тестах, итог тогда с ошибкой.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMEOUT = 900

GROUPS = {
    "rules": {
        "tests": ["tests/test_rules.py"],
        "mutations": [
            ("game/core/game2048.py", "            k += 2\n", "            k += 1\n",
             "плитка сливается дважды за ход"),
            ("game/core/game2048.py", "tiles[k + 1] == e and e < MAX_EXP", "tiles[k + 1] == e",
             "два 32768 сливаются в переполнение"),
            ("game/core/game2048.py", "score += 1 << (e + 1)", "score += 1 << e",
             "очки за слияние вдвое меньше"),
            ("game/core/game2048.py", "np.where(u_val < FOUR_CHANCE, 2, 1)",
             "np.where(u_val < FOUR_CHANCE, 1, 2)", "четвёрки в 90 % случаев"),
            ("game/core/generic.py", "FOUR_CHANCE = 0.1", "FOUR_CHANCE = 0.2",
             "четвёрки в 20 % случаев"),
            ("game/core/game2048.py", "pos = (cum <= k[:, None]).sum(axis=1)",
             "pos = (cum < k[:, None]).sum(axis=1)", "плитка ложится не в ту клетку"),
            ("game/core/game2048.py", "a = a1 | (a2 << U(12)) | (a3 >> U(12))",
             "a = a1 | (a2 << U(12)) | (a3 >> U(8))", "транспонирование портит клетки"),
            ("game/core/game2048.py", "return ~legal.any(axis=1)", "return ~legal.all(axis=1)",
             "конец игры, когда есть ход"),
            ("game/core/game2048.py", "    boards = spawn(boards, rng)\n    return boards,",
             "    return boards,", "новая партия с одной плиткой"),
            ("game/core/game2048.py", "    new[ok] = spawn(new[ok], rng)\n", "",
             "после хода не появляется плитка"),
            ("game/core/generic.py", "tiles[k + 1][1] == v and v < MAX_TILE", "tiles[k + 1][1] == v",
             "образец: два 32768 сливаются"),
            ("game/core/generic.py",
             "    k = int(rng.random() * len(free))\n    value = 4 if rng.random() < FOUR_CHANCE else 2\n",
             "    value = 4 if rng.random() < FOUR_CHANCE else 2\n    k = int(rng.random() * len(free))\n",
             "образец: другой порядок случайных чисел"),
        ],
    },
    "ntuple": {
        "tests": ["tests/test_ntuple.py", "tests/test_td.py"],
        "mutations": [
            ("game/ai/ntuple.py", "return (((b & U(0x000F000F000F000F)) << U(12))",
             "return (((b & U(0x000F000F000F000F)) << U(8))", "отражение портит клетки"),
            ("game/ai/ntuple.py", "flip_v(h), t, th, tv, flip_v(th)])", "flip_v(h), t, th, tv, th])",
             "одна симметрия повторена вместо восьмой"),
            ("game/ai/ntuple.py", "cells[k + 1] == cells[k] + 1", "cells[k + 1] >= cells[k] + 1",
             "клетки не подряд считаются подряд"),
            ("game/ai/ntuple.py", "(lr * total / count)", "(lr * total)",
             "ошибки частых весов складываются, а не усредняются"),
            ("game/ai/ntuple.py", "scale = peak / 32767.0 if", "scale = peak / 255.0 if",
             "квантование переполняет int16"),
            ("game/train/td.py", "target = np.where(dead, 0.0, q[idx, a])",
             "target = np.where(dead, 0.0, reward[idx, a])", "TD без оценки следующей позиции"),
            ("game/train/td.py", "            prev = feats[idx, a]\n", "            prev = feats[idx, 0]\n",
             "учится не та позиция после хода"),
            ("game/train/td.py", "        tr.games_done = state[\"games_done\"]\n", "",
             "продолжение начинает счёт партий с нуля"),
            ("game/train/td.py", "    delta = np.asarray(target, dtype=np.float64) - net.value_of_features(feats)",
             "    delta = np.asarray(target, dtype=np.float64) + net.value_of_features(feats)",
             "TD-ошибка с неверным знаком"),
        ],
    },
    "expectimax": {
        "tests": ["tests/test_expectimax.py"],
        "mutations": [
            ("game/ai/expectimax.py", "weight = np.concatenate([TWO_P * p, FOUR_P * p])",
             "weight = np.concatenate([FOUR_P * p, TWO_P * p])", "вероятности 2 и 4 перепутаны"),
            ("game/ai/expectimax.py", "best[~np.isfinite(best)] = 0.0", "best[~np.isfinite(best)] = 1e6",
             "конец партии стоит не ноль"),
            ("game/ai/expectimax.py", "    p = 1.0 / cnt[rows]\n", "    p = 1.0 + 0 * cnt[rows]\n",
             "пустые клетки не усредняются"),
            ("game/ai/expectimax.py", "v = chance_values(net, afters, depth - 1, chunk)",
             "v = chance_values(net, afters, max(1, depth - 2), chunk)", "глубина считается неверно"),
            ("game/ai/expectimax.py", "vals.ravel()[flat_legal] = reward.ravel()[flat_legal] + v",
             "vals.ravel()[flat_legal] = v", "очки за ход забыты"),
            ("game/ai/expectimax.py", "if depth > 1 and elapsed + last * branching > time_limit:",
             "if False:", "лимит времени не соблюдается"),
            ("game/ai/expectimax.py",
             "weights=weight * best, minlength=part.shape[0])",
             "weights=weight * best, minlength=part.shape[0])[::-1]", "куски путают родителей"),
        ],
    },
    "window": {
        "tests": ["tests/test_app.py", "tests/test_session.py"],
        "mutations": [
            ("game/ui/widget.py", "self.btn_undo.setVisible(mode == \"play\")", "self.btn_undo.setVisible(True)",
             "отмена хода видна во всех режимах"),
            ("game/ui/widget.py", "Session(size, seed=seed), Session(size, seed=seed)",
             "Session(size, seed=seed), Session(size)", "у соревнования разные зёрна"),
            ("game/ui/widget.py",
             ["        if self.paused:\n            self.ai_timer.stop()\n",
              "or self.runner.busy() or self.paused:",
              "elif self.mode == \"ai\" and not self.paused:",
              "if self.mode == \"ai\" and not self.paused and self.ai and not self.ai.over():"],
             ["        if self.paused:\n            pass\n",
              "or self.runner.busy():",
              "elif self.mode == \"ai\":",
              "if self.mode == \"ai\" and self.ai and not self.ai.over():"],
             "пауза не останавливает ИИ"),
            ("game/ui/widget.py", "        self.paused = not self.paused\n", "        self.paused = False\n",
             "кнопка паузы не ставит на паузу"),
            ("game/ui/widget.py", "        self.stepping = True\n        self._ask_ai_move()\n",
             "        self.stepping = True\n", "шаг ничего не делает"),
            ("game/ui/widget.py", "if self.size == 4 and self.runner.brain.available(self.level):", "if True:",
             "«большие в угол» записываются как ИИ"),
            ("game/ui/widget.py", "        return self.runner.stop(2000)", "        return True",
             "закрытие не останавливает поток ИИ"),
            ("game/ui/widget.py", "            self.tr.lang = self.ctx.lang\n", "",
             "смена языка не переводит"),
            ("game/ui/widget.py", "            b.set_theme(self.ctx.theme)", "            pass",
             "смена темы не перекрашивает поле"),
            ("game/ui/widget.py", "                self.me_board.set_hint(result[\"move\"])",
             "                self.me_board.set_hint(None)", "подсказка без стрелки"),
            ("game/ui/board.py", "return 1 if dx > 0 else 3", "return 3 if dx > 0 else 1",
             "смахивание в обратную сторону"),
            ("game/ui/widget.py", "Qt.Key.Key_Left: 3, Qt.Key.Key_A: 3", "Qt.Key.Key_Left: 1, Qt.Key.Key_A: 3",
             "стрелка влево двигает вправо"),
            ("game/ui/widget.py",
             "            if self.ai is not None and not self.ai.over():\n                self._ask_ai_move()\n",
             "            if self.ai is not None and not self.ai.over():\n                pass\n",
             "в соревновании ИИ не отвечает на ход"),
            ("game/ui/widget.py",
             "        result = \"win\" if me > ai else (\"loss\" if me < ai else \"draw\")\n        self.ctx.stats",
             "        result = \"loss\" if me > ai else (\"win\" if me < ai else \"draw\")\n        self.ctx.stats",
             "итог соревнования наоборот"),
            ("game/ui/widget.py", "self.status.setMinimumHeight(28)", "self.status.setFixedHeight(28)",
             "подпись с переносом фиксированной высоты"),
            ("game/ui/widget.py", "        if self.size != 4:\n            notes.append",
             "        if False:\n            notes.append", "о необученном размере молчит"),
            ("game/ui/widget.py", "                w.hide()\n                w.setParent(None)\n", "",
             "старые подписи статистики остаются поверх новых"),
            ("game/core/session.py",
             ["        if self.undo_allowed:\n            self.history.append(snap)",
              "return self.undo_allowed and bool(self.history)"],
             ["        if True:\n            self.history.append(snap)", "return bool(self.history)"],
             "отмена там, где её нет"),
        ],
    },
}


def run_tests(tests):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(var, "2")
    cmd = [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *tests]
    try:
        r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return None
    return r.returncode


def check(group):
    spec = GROUPS[group]
    survived = []
    print(f"== {group}: тесты без мутаций")
    if run_tests(spec["tests"]) != 0:
        print("   тесты красные и без мутаций - проверка бессмысленна")
        return [("(чистый код)", "тесты красные")]
    for path, old, new, what in spec["mutations"]:
        full = os.path.join(ROOT, path)
        with open(full, "rb") as f:
            original = f.read()
        text = original.decode("utf-8")
        # Одна мутация может быть несколькими заменами (списки old/new): где защита стоит
        # в двух местах, убрать одну - значит ничего не сломать.
        olds = old if isinstance(old, list) else [old]
        news = new if isinstance(new, list) else [new]
        if "\r\n" in text:                    # файл с концами строк Windows
            olds = [o.replace("\n", "\r\n") for o in olds]
            news = [n.replace("\n", "\r\n") for n in news]
        bad = [o for o in olds if text.count(o) != 1]
        if bad:
            print(f"   !! мутация «{what}»: строка найдена {text.count(bad[0])} раз, пропуск")
            survived.append((what, "строка не найдена"))
            continue
        mutated = text
        for o, n in zip(olds, news):
            mutated = mutated.replace(o, n)
        started = time.time()
        try:
            with open(full, "wb") as f:
                f.write(mutated.encode("utf-8"))
            code = run_tests(spec["tests"])
        finally:
            with open(full, "wb") as f:
                f.write(original)
        verdict = "поймана" if code not in (0, None) else ("тайм-аут" if code is None else "ВЫЖИЛА")
        print(f"   {verdict:8} {what}  ({time.time() - started:.0f} с)")
        if code == 0:
            survived.append((what, "тесты зелёные"))
    return survived


def main(argv):
    groups = argv or list(GROUPS)
    bad = []
    for g in groups:
        bad += check(g)
    total = sum(len(GROUPS[g]["mutations"]) for g in groups)
    print(f"\nитог: поймано {total - len(bad)} из {total}")
    for what, why in bad:
        print(f"   выжила: {what} - {why}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
