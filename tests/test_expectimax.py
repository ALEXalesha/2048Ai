"""Expectimax: поиск на 1-3 хода с перебором всех новых плиток."""
import time

import numpy as np
import pytest

from game.ai import expectimax as ex
from game.ai import ntuple
from game.ai.ntuple import NTupleNet
from game.core import game2048 as g
from game.core import generic as gen


def b(grid):
    return np.uint64(g.encode(grid))


def constant_net(c, tuples=ntuple.TUPLES_4):
    """Сеть, у которой любая позиция стоит c: проверяет поиск, а не оценку."""
    net = NTupleNet(tuples)
    net.w[:] = c / net.n_features
    return net


def random_net(seed=0):
    net = NTupleNet(ntuple.TUPLES_4)
    rng = np.random.default_rng(seed)
    net.w[:] = rng.normal(scale=50, size=net.w.size).astype(np.float32)
    return net


# --- медленный образец: рекурсия по правилам generic ----------------------------------

def slow_value(net, grid):
    return float(net.value(np.array([g.encode(grid)], dtype=np.uint64))[0])


def slow_action_values(net, grid, depth):
    out = []
    for d in gen.DIRECTIONS:
        new, reward, moved, _ = gen.slide(grid, d)
        if not moved:
            out.append(-np.inf)
            continue
        if depth <= 1:
            out.append(reward + slow_value(net, new))
            continue
        free = gen.empty_cells(new)
        total = 0.0
        for r, c in free:
            for tile, p in ((2, 0.9), (4, 0.1)):
                child = [row[:] for row in new]
                child[r][c] = tile
                best = max(slow_action_values(net, child, depth - 1))
                total += p / len(free) * (best if np.isfinite(best) else 0.0)
        out.append(reward + total)
    return out


def random_positions(n, seed, fill=(8, 13)):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        grid = [[0] * 4 for _ in range(4)]
        k = int(rng.integers(*fill))
        cells = rng.choice(16, size=k, replace=False)
        for c in cells:
            grid[c // 4][c % 4] = int(2 ** rng.integers(1, 8))
        out.append(grid)
    return out


@pytest.mark.parametrize("depth", [1, 2])
def test_matches_slow_reference(depth):
    net = random_net(1)
    grids = random_positions(12, seed=depth)
    boards = np.array([g.encode(x) for x in grids], dtype=np.uint64)
    fast = ex.action_values(net, boards, depth)
    for k, grid in enumerate(grids):
        slow = slow_action_values(net, grid, depth)
        assert np.allclose(fast[k], slow, rtol=1e-6, atol=1e-3), (grid, fast[k], slow)


def test_depth_three_matches_slow_reference_on_crowded_boards():
    net = random_net(2)
    grids = random_positions(3, seed=9, fill=(12, 14))
    boards = np.array([g.encode(x) for x in grids], dtype=np.uint64)
    fast = ex.action_values(net, boards, 3)
    for k, grid in enumerate(grids):
        assert np.allclose(fast[k], slow_action_values(net, grid, 3), rtol=1e-6, atol=1e-3)


def test_chunking_does_not_change_the_answer():
    net = random_net(3)
    boards = np.array([g.encode(x) for x in random_positions(20, seed=4)], dtype=np.uint64)
    whole = ex.action_values(net, boards, 2)
    small = ex.action_values(net, boards, 2, chunk=37)
    assert np.allclose(whole, small)


# --- очевидные ходы -----------------------------------------------------------------

def test_takes_the_merge_into_2048():
    grid = [[1024, 1024, 0, 0],
            [4, 8, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]
    net = NTupleNet(ntuple.TUPLES_4)
    for depth in (1, 2, 3):
        move = ex.best_move(net, b(grid), depth=depth)
        assert move in (g.LEFT, g.RIGHT), depth


# Позиция найдена перебором: «жадный» ход сразу даёт очки, но после любой новой плитки
# ходов больше нет; другой ход очков не даёт, зато партия живёт.
DEATH_TRAP = [[2, 4, 8, 16],
              [4, 8, 16, 32],
              [8, 16, 32, 64],
              [16, 32, 64, 64]]


def test_search_avoids_a_move_that_loses_the_game():
    net = constant_net(1e6)
    board = b(DEATH_TRAP)
    q1 = ex.action_values(net, np.array([board]), 1)[0]
    q2 = ex.action_values(net, np.array([board]), 2)[0]
    # Без поиска все разрешённые ходы стоят одинаково (+очки), с поиском смертельный
    # ход проваливается почти до нуля.
    assert np.isfinite(q1).sum() >= 2
    worst = np.nanargmin(np.where(np.isfinite(q2), q2, np.nan))
    assert q2[worst] < 0.5 * q2.max()
    assert ex.best_move(net, board, depth=2) != worst


def test_no_moves_means_no_best_move():
    stuck = b([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    net = NTupleNet(ntuple.TUPLES_4)
    assert ex.best_move(net, stuck, depth=2) is None
    assert np.isneginf(ex.action_values(net, np.array([stuck]), 2)).all()


def test_time_limit_is_respected_and_still_answers():
    net = NTupleNet(ntuple.TUPLES_6)
    board = b([[2, 0, 0, 0], [0, 4, 0, 0], [0, 0, 8, 0], [0, 0, 0, 0]])
    started = time.time()
    res = ex.search(net, board, max_depth=3, time_limit=0.05)
    took = time.time() - started
    assert res.move in range(4)
    assert 1 <= res.depth <= 3
    assert took < 0.05 + 1.0          # прерывается между глубинами, а не посреди
    # Совсем без времени - только глубина 1, но ответ всё равно есть.
    quick = ex.search(net, board, max_depth=3, time_limit=0.0)
    assert quick.depth == 1 and quick.move in range(4)


def test_search_reports_values_for_every_direction():
    net = constant_net(100.0)
    board = b([[2, 2, 0, 0], [0] * 4, [0] * 4, [0] * 4])
    res = ex.search(net, board, max_depth=2, time_limit=5)
    assert res.depth == 2
    assert len(res.values) == 4
    assert res.move == int(np.argmax(res.values))
