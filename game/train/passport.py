"""Паспорт модели: проверки силы по spec §4.

По 1000 партий на уровень и на каждого простого бота, у партии i своё зерно SEED + i.
Пишется доля партий с плитками 2048...32768, средний и лучший счёт, время, и счёт первых
партий - тесты пересчитывают их с теми же зёрнами и сверяют до очка.

    python -m game.train passport --run <папка обучения> [--games 1000]

Готовые части сохраняются сразу: прерванный прогон продолжается с недостающих.
"""
from __future__ import annotations

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


def play_parallel(name, games, processes, echo=print):
    from concurrent.futures import ProcessPoolExecutor
    size = max(1, -(-games // (processes * 4)))
    jobs = [(name, lo, min(games, lo + size)) for lo in range(0, games, size)]
    parts = {}
    with ProcessPoolExecutor(max_workers=processes) as pool:
        for lo, res in pool.map(_chunk, jobs):
            parts[lo] = res
            if echo:
                echo(f"  {name}: готово {sum(len(r['score']) for r in parts.values())} из {games}")
    keys = sorted(parts)
    out = {k: np.concatenate([parts[lo][k] for lo in keys]) for k in ("score", "max_exp", "moves")}
    out["seconds"] = sum(parts[lo]["seconds"] for lo in keys)
    return out


def players(brain):
    out = {"random": lambda: bots.random_policy(SEED),
           "corner": bots.corner_policy}
    for level in levels.LEVELS:
        out[level] = (lambda lv=level: levels.policy(brain, lv))
    return out


def evaluate(name, policy, games, echo=print, processes=1):
    started = time.time()
    last = [0.0]

    def progress(lo, hi, alive, steps):
        if echo and time.time() - last[0] > 60:
            last[0] = time.time()
            echo(f"  {name}: партии {lo}-{hi}, идут {alive}, ход {steps}, {time.time() - started:.0f} с")

    if processes > 1 and name in HEAVY:
        res = play_parallel(name, games, processes, echo)
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


def build(out_path, run_dir=None, games=1000, only=None, echo=print, extra=None, processes=1):
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
        passport["results"][name] = evaluate(name, make(), games, echo, processes)
        r = passport["results"][name]
        echo(f"{name}: средний {r['avg_score']:.0f}, 2048 {100 * r['reached']['2048']:.1f}%, "
             f"{r['seconds']:.0f} с")
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(passport, f, ensure_ascii=False, indent=1)
        os.replace(tmp, out_path)
    return passport
