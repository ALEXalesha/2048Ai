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
SAMPLE = {"random": 50, "corner": 50, "novice": 20, "medium": 20, "strong": 3}


def players(brain):
    out = {"random": lambda: bots.random_policy(SEED),
           "corner": bots.corner_policy}
    for level in levels.LEVELS:
        out[level] = (lambda lv=level: levels.policy(brain, lv))
    return out


def evaluate(name, policy, games, echo=print):
    started = time.time()
    last = [0.0]

    def progress(lo, hi, alive, steps):
        if echo and time.time() - last[0] > 60:
            last[0] = time.time()
            echo(f"  {name}: партии {lo}-{hi}, идут {alive}, ход {steps}, {time.time() - started:.0f} с")

    res = bots.play(policy, games, SEED, progress=progress)
    s = bots.summary(res)
    s["seconds"] = round(time.time() - started, 1)
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


def build(out_path, run_dir=None, games=1000, only=None, echo=print, extra=None):
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
        passport["results"][name] = evaluate(name, make(), games, echo)
        r = passport["results"][name]
        echo(f"{name}: средний {r['avg_score']:.0f}, 2048 {100 * r['reached']['2048']:.1f}%, "
             f"{r['seconds']:.0f} с")
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(passport, f, ensure_ascii=False, indent=1)
        os.replace(tmp, out_path)
    return passport
