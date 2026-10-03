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
        if "\r\n" in text:                    # файл с концами строк Windows
            old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
        if text.count(old) != 1:
            print(f"   !! мутация «{what}»: строка найдена {text.count(old)} раз, пропуск")
            survived.append((what, "строка не найдена"))
            continue
        started = time.time()
        try:
            with open(full, "wb") as f:
                f.write(text.replace(old, new).encode("utf-8"))
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
