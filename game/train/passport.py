"""Паспорт модели: проверки силы по spec §4.

По 1000 партий на уровень и на каждого простого бота, у партии i своё зерно SEED + i.
Пишется доля партий с плитками 2048...32768, средний и лучший счёт, время, и счёт первых
партий - тесты пересчитывают их с теми же зёрнами и сверяют до очка.

    python -m game.train passport --run <папка обучения> [--games 1000]

Готовые части сохраняются сразу: прерванный прогон продолжается с недостающих.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time

import numpy as np

from game.ai import bots, levels
from game.ai.levels import LEVEL_SETTINGS, Brain

SEED = 20261003
SAMPLE = {"random": 50, "corner": 50, "novice": 20, "medium": 20, "strong": 2}
# Сильный уровень с поиском в 19 раз медленнее сети без поиска: его партии делятся между
# несколькими процессами, каждый в один поток и с пониженным приоритетом.
HEAVY = ("strong",)
# По столько партий в куске сильного уровня: пачка побольше быстрее считается, а прерывание
# теряет не больше одного куска на процесс.
PART_GAMES = 25


def _chunk(job):
    """Партии [lo, hi) одного игрока в отдельном процессе. Зерно партии - SEED + её номер,
    как и в общем прогоне, поэтому куски складываются в тот же результат."""
    name, lo, hi = job
    from game.train.td import limit_threads, lower_priority
    limit_threads(1)
    lower_priority()
    policy = players(Brain())[name]()
    res = bots.play(policy, hi - lo, SEED + lo)
    return lo, res


FIELDS = ("score", "max_exp", "moves")


def fingerprint(name):
    """Чем играл кусок: модель уровня (её содержимое), настройки уровня и зерно. Куски
    другой модели в новый паспорт не берутся."""
    h = hashlib.sha1(f"{name}|{SEED}".encode())
    s = LEVEL_SETTINGS.get(name)
    if s:
        h.update(json.dumps(s, sort_keys=True).encode())
        path = levels.find_model(s["model"])
        if path:
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
    return h.hexdigest()[:12]


def _part_path(parts_dir, name, lo, hi, key):
    return os.path.join(parts_dir, f"{name}-{key}-{lo:05d}-{hi:05d}.npz")


def _save_part(path, res):
    tmp = path + ".tmp.npz"
    np.savez(tmp, seconds=np.array(res["seconds"]), **{k: np.asarray(res[k]) for k in FIELDS})
    os.replace(tmp, path)


def _load_part(path):
    with np.load(path) as z:
        out = {k: z[k] for k in FIELDS}
        out["seconds"] = float(z["seconds"])
    return out


def play_parallel(name, games, processes, echo=print, parts_dir=None, chunk=None, limit_chunks=None):
    """Партии игрока кусками в нескольких процессах. С parts_dir каждый готовый кусок сразу
    ложится файлом, и прерванный прогон продолжается с недостающих. limit_chunks - досчитать
    не больше стольких кусков (для проверки продолжения). Возвращает None, если не всё готово."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    size = chunk or max(1, -(-games // (processes * 4)))
    spans = [(lo, min(games, lo + size)) for lo in range(0, games, size)]
    key = fingerprint(name)
    parts = {}
    todo = []
    for lo, hi in spans:
        path = _part_path(parts_dir, name, lo, hi, key) if parts_dir else None
        if path and os.path.exists(path):
            parts[lo] = _load_part(path)
        else:
            todo.append((lo, hi, path))
    if parts and echo:
        echo(f"  {name}: из прошлых прогонов {sum(len(r['score']) for r in parts.values())} из {games}")
    if limit_chunks is not None:
        todo = todo[:limit_chunks]
    if parts_dir:
        os.makedirs(parts_dir, exist_ok=True)
    if todo:
        with ProcessPoolExecutor(max_workers=processes) as pool:
            futures = {pool.submit(_chunk, (name, lo, hi)): path for lo, hi, path in todo}
            for fut in as_completed(futures):
                lo, res = fut.result()
                parts[lo] = res
                if futures[fut]:
                    _save_part(futures[fut], res)
                if echo:
                    echo(f"  {name}: готово {sum(len(r['score']) for r in parts.values())} из {games}")
    if len(parts) < len(spans):
        return None
    keys = sorted(parts)
    out = {k: np.concatenate([parts[lo][k] for lo in keys]) for k in FIELDS}
    out["seconds"] = sum(parts[lo]["seconds"] for lo in keys)
    return out


def players(brain):
    out = {"random": lambda: bots.random_policy(SEED),
           "corner": bots.corner_policy}
    for level in levels.LEVELS:
        out[level] = (lambda lv=level: levels.policy(brain, lv))
    return out


def evaluate(name, policy, games, echo=print, processes=1, parts_dir=None):
    started = time.time()
    last = [0.0]

    def progress(lo, hi, alive, steps):
        if echo and time.time() - last[0] > 60:
            last[0] = time.time()
            echo(f"  {name}: партии {lo}-{hi}, идут {alive}, ход {steps}, {time.time() - started:.0f} с")

    if name in HEAVY:
        res = play_parallel(name, games, processes, echo, parts_dir=parts_dir, chunk=PART_GAMES)
        if res is None:
            raise RuntimeError(f"{name}: не все куски готовы")
    else:
        res = bots.play(policy, games, SEED, progress=progress)
    s = bots.summary(res)
    s["seconds"] = round(time.time() - started, 1)
    s["processes"] = processes if name in HEAVY else 1
    s["seed"] = SEED
    s["first_scores"] = [int(x) for x in res["score"][:SAMPLE.get(name, 10)]]
    s["max_tile_counts"] = {str(1 << int(e)): int((res["max_exp"] == e).sum())
                            for e in np.unique(res["max_exp"])}
    return s


def training_info(run_dir):
    with open(os.path.join(run_dir, "state.json"), encoding="utf-8") as f:
        state = json.load(f)
    return {
        "games": state["games_done"], "moves": state["moves_done"],
        "hours": round(state["seconds"] / 3600, 2), "batch": state["batch"],
        "final_alpha": state["alpha"], "tuples": state["tuples"],
        "threads": 1, "cpu": platform.processor() or platform.machine(),
    }


def build(out_path, run_dir=None, games=1000, only=None, echo=print, extra=None, processes=1,
          parts_dir=None):
    passport = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            passport = json.load(f)
    passport.setdefault("eval", {"games": games, "seed": SEED})
    if passport["eval"].get("games") != games:
        passport = {"eval": {"games": games, "seed": SEED}}
    if run_dir:
        passport["training"] = training_info(run_dir)
    passport["level_settings"] = LEVEL_SETTINGS
    if extra:
        passport.update(extra)
    passport.setdefault("results", {})
    brain = Brain()
    for name, make in players(brain).items():
        if only and name not in only:
            continue
        if name in passport["results"]:
            continue
        echo(f"{name}: {games} партий...")
        passport["results"][name] = evaluate(name, make(), games, echo, processes, parts_dir)
        r = passport["results"][name]
        echo(f"{name}: средний {r['avg_score']:.0f}, 2048 {100 * r['reached']['2048']:.1f}%, "
             f"{r['seconds']:.0f} с")
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(passport, f, ensure_ascii=False, indent=1)
        os.replace(tmp, out_path)
    return passport
