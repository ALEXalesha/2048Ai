"""Стратегии хода для быстрой доски и прогон партий пачкой.

Стратегия - функция: поля (N,) -> ходы (N,). Ходы выбираются только из разрешённых;
если ходов нет, партия уже кончилась и ход не важен.
"""
from __future__ import annotations

import time

import numpy as np

from game.core import game2048 as g


def _mix(x):
    """splitmix64: из числа - «случайное» число, всегда одно и то же."""
    x = (x + np.uint64(0x9E3779B97F4A7C15))
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def random_policy(seed=0):
    """Случайный разрешённый ход. Выбор зависит только от поля и зерна (хеш), а не от
    того, какие ещё партии идут в пачке: партия i играется одинаково при любой пачке -
    так паспорт пересчитывается на малой выборке."""
    salt = np.uint64(int(seed) & 0xFFFFFFFFFFFFFFFF)

    def policy(boards):
        _, _, legal = g.all_moves(boards)
        keys = _mix(np.asarray(boards, dtype=np.uint64)[:, None] ^ salt
                    ^ (np.arange(4, dtype=np.uint64)[None, :] * np.uint64(0x632BE59BD9B4E019)))
        noise = (keys >> np.uint64(11)).astype(np.float64)
        noise[~legal] = -1.0
        return noise.argmax(axis=1)
    policy.name = "random"
    return policy


# «Держи большие в углу»: вниз, влево, вправо, вверх - первый разрешённый.
CORNER_ORDER = (g.DOWN, g.LEFT, g.RIGHT, g.UP)


def corner_policy():
    order = np.array(CORNER_ORDER)
    rank = np.empty(4, dtype=np.int64)
    rank[order] = np.arange(4)

    def policy(boards):
        _, _, legal = g.all_moves(boards)
        score = np.where(legal, 10 - rank[None, :], -1)
        return score.argmax(axis=1)
    policy.name = "corner"
    return policy


def greedy_values(net, boards):
    """Оценка каждого хода без поиска: очки хода + оценка сети после хода. (N, 4),
    у запрещённых ходов -inf."""
    after, reward, legal = g.all_moves(boards)
    vals = net.value(after.ravel()).reshape(after.shape)
    q = reward + vals
    q[~legal] = -np.inf
    return q


def greedy_policy(net):
    def policy(boards):
        return greedy_values(net, boards).argmax(axis=1)
    policy.name = "greedy"
    return policy


def new_board(rng):
    """Начальное поле партии из её генератора - то же, что у generic.Game с тем же зерном."""
    return g.new_games(1, rng)[0][0]


def play(policy, n, seed, batch=1000, max_moves=200_000, progress=None):
    """Сыграть n партий стратегией. Зерно задаёт и начальные поля, и все новые плитки:
    партия i всегда получает один и тот же поток случайности (своё зерно seed + i), поэтому
    результат не зависит от размера пачки. Возвращает словарь массивов: score, max_exp,
    moves и seconds."""
    scores = np.zeros(n, dtype=np.int64)
    max_exp = np.zeros(n, dtype=np.int64)
    moves = np.zeros(n, dtype=np.int64)
    started = time.time()
    for lo in range(0, n, batch):
        hi = min(n, lo + batch)
        rngs = [np.random.default_rng(seed + i) for i in range(lo, hi)]
        boards = np.empty(hi - lo, dtype=np.uint64)
        for k, r in enumerate(rngs):
            boards[k] = new_board(r)
        sc = np.zeros(hi - lo, dtype=np.int64)
        mv = np.zeros(hi - lo, dtype=np.int64)
        alive = ~g.game_over(boards)
        steps = 0
        while alive.any() and steps < max_moves:
            idx = np.flatnonzero(alive)
            b = boards[idx]
            a = np.asarray(policy(b))
            after, reward, legal = g.all_moves(b)
            j = np.arange(len(idx))
            ok = legal[j, a]
            new = after[j, a][ok]
            go = idx[ok]
            # Новая плитка - из генератора своей партии: так партии независимы от пачки.
            u = np.array([rngs[i].random(2) for i in go]).reshape(-1, 2)
            boards[go] = g.spawn_with(new, u[:, 0], u[:, 1])
            sc[go] += reward[j, a][ok]
            mv[go] += 1
            alive[idx] = ~g.game_over(boards[idx])
            steps += 1
            if progress is not None:
                progress(lo, hi, int(alive.sum()), steps)
        scores[lo:hi] = sc
        max_exp[lo:hi] = g.max_tile(boards)
        moves[lo:hi] = mv
    return {"score": scores, "max_exp": max_exp, "moves": moves, "seconds": time.time() - started}


def summary(result, tiles=(2048, 4096, 8192, 16384, 32768)):
    """Цифры для паспорта: доли партий с плиткой, средний и лучший счёт."""
    score = result["score"]
    max_exp = result["max_exp"]
    out = {
        "games": int(len(score)),
        "avg_score": float(score.mean()) if len(score) else 0.0,
        "best_score": int(score.max()) if len(score) else 0,
        "avg_moves": float(result["moves"].mean()) if len(score) else 0.0,
        "reached": {},
    }
    for t in tiles:
        e = int(t).bit_length() - 1
        out["reached"][str(t)] = float((max_exp >= e).mean()) if len(score) else 0.0
    return out
