"""Expectimax: сеть плюс поиск на 1-3 хода вперёд.

Глубина - сколько своих ходов просматривается. Глубина 1 - это просто сеть: очки хода
плюс её оценка позиции после хода. На глубине 2 после каждого хода перебираются все
пустые клетки и обе новые плитки (2 с вероятностью 0.9, 4 - 0.1), в каждой такой позиции
берётся лучший следующий ход, и всё складывается со своими вероятностями. Позиция без
ходов стоит 0 - так поиск видит смертельные ходы, которых сеть может не заметить.

Перебор идёт пачкой: все листья дерева оцениваются сетью одной выборкой из таблиц.
Чтобы не съесть память, листья режутся на куски (chunk).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from game.core import game2048 as g

U = np.uint64
TWO_P, FOUR_P = 1.0 - g.FOUR_CHANCE, g.FOUR_CHANCE
CHUNK = 60_000


def expand(afters):
    """Все варианты новой плитки: (дети, номер родителя, вероятность)."""
    empty = g.cells(afters) == 0
    rows, cols = np.nonzero(empty)
    cnt = empty.sum(axis=1)
    base = afters[rows]
    shift = cols.astype(np.uint64) * U(4)
    children = np.concatenate([base | (U(1) << shift), base | (U(2) << shift)])
    parent = np.concatenate([rows, rows])
    p = 1.0 / cnt[rows]
    weight = np.concatenate([TWO_P * p, FOUR_P * p])
    return children, parent, weight


def chance_values(net, afters, depth, chunk=CHUNK):
    """Ожидаемая стоимость позиций после хода, когда дальше depth своих ходов."""
    out = np.zeros(afters.shape[0], dtype=np.float64)
    if afters.shape[0] == 0:
        return out
    # Кусок родителей подбирается так, чтобы детей было не больше chunk.
    step = max(1, chunk // 32)
    for lo in range(0, afters.shape[0], step):
        part = afters[lo:lo + step]
        children, parent, weight = expand(part)
        best = action_values(net, children, depth, chunk).max(axis=1)
        best[~np.isfinite(best)] = 0.0          # ходов нет - партия кончилась
        out[lo:lo + step] = np.bincount(parent, weights=weight * best, minlength=part.shape[0])
    return out


def action_values(net, boards, depth, chunk=CHUNK):
    """Стоимость каждого из четырёх ходов (UP, RIGHT, DOWN, LEFT): массив (N, 4),
    у запрещённых ходов -inf."""
    boards = np.asarray(boards, dtype=np.uint64)
    n = boards.shape[0]
    vals = np.full((n, 4), -np.inf)
    if n == 0:
        return vals
    after, reward, legal = g.all_moves(boards)
    flat_legal = np.flatnonzero(legal.ravel())
    afters = after.ravel()[flat_legal]
    if depth <= 1:
        v = np.empty(afters.shape[0], dtype=np.float64)
        for lo in range(0, afters.shape[0], chunk):
            v[lo:lo + chunk] = net.value(afters[lo:lo + chunk])
    else:
        v = chance_values(net, afters, depth - 1, chunk)
    vals.ravel()[flat_legal] = reward.ravel()[flat_legal] + v
    return vals


def best_move(net, board, depth=2):
    """Лучший ход или None, если ходов нет."""
    q = action_values(net, np.array([board], dtype=np.uint64), depth)[0]
    if not np.isfinite(q).any():
        return None
    return int(np.argmax(q))


def adaptive_depth(boards, base=2, deep_below=4, max_depth=3):
    """Глубина по тесноте поля: когда пустых клеток мало, ветвей мало и опасность
    близко - смотрим на ход дальше."""
    empty = g.count_empty(boards)
    return np.where(empty <= deep_below, np.minimum(base + 1, max_depth), base)


def policy(net, depth=2, adaptive=False, deep_below=4):
    """Стратегия для bots.play: ход по поиску заданной глубины."""
    def choose(boards):
        if not adaptive:
            return action_values(net, boards, depth).argmax(axis=1)
        d = adaptive_depth(boards, base=depth, deep_below=deep_below)
        out = np.zeros(boards.shape[0], dtype=np.int64)
        for dd in np.unique(d):
            m = d == dd
            out[m] = action_values(net, boards[m], int(dd)).argmax(axis=1)
        return out
    choose.name = f"expectimax{depth}{'+' if adaptive else ''}"
    return choose


@dataclass
class SearchResult:
    move: int | None
    values: list = field(default_factory=list)   # по направлениям UP, RIGHT, DOWN, LEFT
    depth: int = 0
    seconds: float = 0.0


def search(net, board, max_depth=3, time_limit=0.3):
    """Поиск с лимитом времени: глубина 1, 2, 3... пока следующая успевает.

    Время следующей глубины прикидывается по прошлой: ветвление - 4 хода на 2 плитки на
    каждую пустую клетку. Если не успеть - остаётся ответ прошлой глубины."""
    started = time.time()
    board = np.array([board], dtype=np.uint64)
    best = SearchResult(None, [-np.inf] * 4, 0, 0.0)
    branching = 4 * 2 * max(1, int(g.count_empty(board)[0]))
    last = 0.0
    for depth in range(1, max_depth + 1):
        elapsed = time.time() - started
        if depth > 1 and elapsed + last * branching > time_limit:
            break
        t0 = time.time()
        q = action_values(net, board, depth)[0]
        last = time.time() - t0
        move = int(np.argmax(q)) if np.isfinite(q).any() else None
        best = SearchResult(move, [float(x) for x in q], depth, time.time() - started)
        if move is None:
            break
    return best
