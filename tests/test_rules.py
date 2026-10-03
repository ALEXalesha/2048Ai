"""Правила 2048: быстрая доска в 64-битном числе (game2048) и простая медленная доска
любого размера (generic) - вторая служит образцом для первой."""
import numpy as np
import pytest

from game.core import game2048 as g
from game.core import generic as gen

UP, RIGHT, DOWN, LEFT = g.UP, g.RIGHT, g.DOWN, g.LEFT


def one(grid, direction):
    """Сдвиг одной доски 4x4: (новая сетка, очки, сдвинулось ли)."""
    b = np.array([g.encode(grid)], dtype=np.uint64)
    nb, reward = g.move(b, direction)
    return g.decode(nb[0]).tolist(), int(reward[0]), bool(nb[0] != b[0])


def row(cells, direction=LEFT):
    grid = [cells, [0] * 4, [0] * 4, [0] * 4]
    out, reward, _ = one(grid, direction)
    return out[0], reward


# --- кодирование ---------------------------------------------------------------------

def test_encode_decode_round_trip():
    grid = [[0, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2048], [4096, 8192, 16384, 32768]]
    assert g.decode(g.encode(grid)).tolist() == grid


def test_cell_layout_is_four_bits_per_cell_row_major():
    grid = [[2, 0, 0, 0], [0] * 4, [0] * 4, [0] * 4]
    assert g.encode(grid) == 1
    grid = [[0] * 4, [0, 4, 0, 0], [0] * 4, [0] * 4]
    assert g.encode(grid) == 2 << (4 * 5)


def test_transpose_swaps_rows_and_columns():
    rng = np.random.default_rng(1)
    grids = rng.integers(0, 16, size=(200, 4, 4))
    boards = np.array([g.encode_exps(x) for x in grids], dtype=np.uint64)
    t = g.transpose(boards)
    for x, b in zip(grids, t):
        assert (g.exps(b) == x.T).all()


# --- сдвиг и слияние ------------------------------------------------------------------

@pytest.mark.parametrize("cells, expected, score", [
    ([2, 2, 0, 0], [4, 0, 0, 0], 4),
    ([0, 2, 0, 2], [4, 0, 0, 0], 4),
    ([2, 2, 2, 2], [4, 4, 0, 0], 8),
    ([4, 4, 8, 0], [8, 8, 0, 0], 8),        # 8 из слияния не сливается с 8 в тот же ход
    ([2, 2, 4, 8], [4, 4, 8, 0], 4),
    ([8, 4, 4, 0], [8, 8, 0, 0], 8),
    ([2, 2, 4, 4], [4, 8, 0, 0], 12),
    ([2, 4, 8, 16], [2, 4, 8, 16], 0),
    ([0, 0, 0, 2], [2, 0, 0, 0], 0),
    ([2, 0, 2, 4], [4, 4, 0, 0], 4),
    ([1024, 1024, 0, 0], [2048, 0, 0, 0], 2048),
])
def test_slide_left(cells, expected, score):
    assert row(cells) == (expected, score)


def test_each_tile_merges_at_most_once_per_move():
    # Во всех строках, где после сдвига плитки не стало больше, чем пар: сумма плиток
    # сохраняется, а число слияний не больше половины плиток.
    for cells, out in [([2, 2, 2, 2], [4, 4, 0, 0]), ([4, 2, 2, 0], [4, 4, 0, 0]),
                       ([4, 4, 4, 0], [8, 4, 0, 0]), ([8, 8, 16, 0], [16, 16, 0, 0])]:
        assert row(cells)[0] == out


def test_right_up_down():
    grid = [[2, 2, 0, 0],
            [0, 0, 0, 0],
            [2, 0, 0, 4],
            [0, 0, 0, 4]]
    assert one(grid, RIGHT)[0] == [[0, 0, 0, 4], [0, 0, 0, 0], [0, 0, 2, 4], [0, 0, 0, 4]]
    assert one(grid, UP)[0] == [[4, 2, 0, 8], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    assert one(grid, DOWN)[0] == [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [4, 2, 0, 8]]
    assert one(grid, LEFT)[0] == [[4, 0, 0, 0], [0, 0, 0, 0], [2, 4, 0, 0], [4, 0, 0, 0]]
    assert one(grid, UP)[1] == 12


def test_move_that_changes_nothing_is_illegal():
    grid = [[2, 4, 8, 16], [0] * 4, [0] * 4, [0] * 4]
    assert one(grid, LEFT)[2] is False
    assert one(grid, RIGHT)[2] is False
    assert one(grid, DOWN)[2] is True


def test_biggest_tile_does_not_overflow():
    # 32768 - предел четырёх бит; два таких не сливаются (65536 не во что записать).
    assert row([32768, 32768, 0, 0]) == ([32768, 32768, 0, 0], 0)
    assert row([16384, 16384, 0, 0]) == ([32768, 0, 0, 0], 32768)


def test_every_row_matches_the_slow_reference():
    for r in range(1 << 16):
        cells = [(r >> (4 * c)) & 0xF for c in range(4)]
        values = [0 if e == 0 else 1 << e for e in cells]
        ref, score, _ = gen.slide_line(values)
        out = int(g.ROW_LEFT[r])
        got = [(out >> (4 * c)) & 0xF for c in range(4)]
        assert [0 if e == 0 else 1 << e for e in got] == ref, values
        assert int(g.ROW_SCORE[r]) == score, values


# --- новая плитка ---------------------------------------------------------------------

def test_spawn_is_90_percent_twos_and_10_percent_fours():
    rng = np.random.default_rng(42)
    boards = np.zeros(200_000, dtype=np.uint64)
    out = g.spawn(boards, rng)
    vals = np.array([g.exps(b).max() for b in out[:20000]])
    assert set(np.unique(vals)) == {1, 2}
    # Доля четвёрок по всей выборке - через счёт по битам, без цикла.
    nib = (out[:, None] >> (np.arange(16, dtype=np.uint64) * np.uint64(4))) & np.uint64(0xF)
    fours = (nib == 2).any(axis=1).mean()
    assert abs(fours - 0.10) < 0.004, fours


def test_spawn_goes_to_an_empty_cell_uniformly():
    rng = np.random.default_rng(3)
    grid = [[2, 0, 4, 0], [0, 8, 0, 0], [0, 0, 0, 16], [32, 0, 0, 0]]
    base = g.encode(grid)
    boards = np.full(80_000, base, dtype=np.uint64)
    out = g.spawn(boards, rng)
    empty = [(r, c) for r in range(4) for c in range(4) if grid[r][c] == 0]
    counts = {cell: 0 for cell in empty}
    diff = out ^ np.uint64(base)
    for i in range(16):
        hit = ((diff >> np.uint64(4 * i)) & np.uint64(0xF)) != 0
        n = int(hit.sum())
        cell = (i // 4, i % 4)
        if cell in counts:
            counts[cell] = n
        else:
            assert n == 0, f"плитка легла на занятую клетку {cell}"
    expected = len(boards) / len(empty)
    for cell, n in counts.items():
        assert abs(n - expected) < 0.06 * expected, (cell, n, expected)
    # Ровно одна новая плитка.
    added = np.zeros(len(boards), dtype=int)
    for i in range(16):
        added += (((diff >> np.uint64(4 * i)) & np.uint64(0xF)) != 0)
    assert (added == 1).all()


def test_spawn_leaves_full_board_alone():
    full = g.encode([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    out = g.spawn(np.array([full], dtype=np.uint64), np.random.default_rng(0))
    assert out[0] == full


def test_new_game_has_two_tiles():
    boards, scores, alive = g.new_games(500, np.random.default_rng(5))
    for b in boards:
        e = g.exps(b)
        assert np.count_nonzero(e) == 2
        assert set(e[e > 0]) <= {1, 2}
    assert (scores == 0).all() and alive.all()


# --- конец игры -----------------------------------------------------------------------

def test_game_over():
    stuck = g.encode([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    merge_h = g.encode([[2, 2, 8, 4], [4, 8, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    merge_v = g.encode([[2, 4, 2, 4], [2, 8, 4, 2], [8, 4, 2, 4], [4, 2, 4, 2]])
    with_hole = g.encode([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 0, 4], [4, 2, 4, 2]])
    over = g.game_over(np.array([stuck, merge_h, merge_v, with_hole], dtype=np.uint64))
    assert over.tolist() == [True, False, False, False]


def test_play_step_ignores_illegal_move_and_spawns_after_legal():
    rng = np.random.default_rng(0)
    b = g.encode([[2, 4, 8, 16], [0] * 4, [0] * 4, [0] * 4])
    boards = np.array([b, b], dtype=np.uint64)
    scores = np.zeros(2, dtype=np.int64)
    alive = np.ones(2, dtype=bool)
    nb, ns, na = g.play_step(boards, scores, alive, np.array([LEFT, DOWN]), rng)
    assert nb[0] == b                                    # нельзя - ничего не меняется
    assert np.count_nonzero(g.exps(nb[1])) == 5          # 4 плитки сдвинулись + 1 новая
    assert ns.tolist() == [0, 0] and na.all()


def test_dead_games_stay_frozen():
    rng = np.random.default_rng(0)
    stuck = g.encode([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    nb, ns, na = g.play_step(np.array([stuck], dtype=np.uint64), np.array([10]),
                             np.array([False]), np.array([UP]), rng)
    assert nb[0] == stuck and ns[0] == 10 and not na[0]


# --- сравнение с медленной простой реализацией на случайных партиях ---------------------

def test_random_games_match_the_slow_reference():
    for seed in range(40):
        rng_fast = np.random.default_rng(seed)
        rng_slow = np.random.default_rng(seed)
        boards, scores, alive = g.new_games(1, rng_fast)
        ref = gen.Game(4, rng_slow)
        assert g.decode(boards[0]).tolist() == ref.grid
        pick = np.random.default_rng(1000 + seed)
        steps = 0
        while alive[0]:
            d = int(pick.integers(0, 4))
            boards, scores, alive = g.play_step(boards, scores, alive, np.array([d]), rng_fast)
            ref.move(d)
            steps += 1
            assert g.decode(boards[0]).tolist() == ref.grid, (seed, steps)
            assert int(scores[0]) == ref.score
            assert bool(alive[0]) == (not ref.over())
        assert steps > 20


def test_all_moves_agrees_with_single_moves():
    rng = np.random.default_rng(9)
    boards = rng.integers(0, 2 ** 63, size=300, dtype=np.int64).astype(np.uint64)
    after, reward, legal = g.all_moves(boards)
    for d in range(4):
        nb, r = g.move(boards, d)
        assert (after[:, d] == nb).all()
        assert (reward[:, d] == r).all()
        assert (legal[:, d] == (nb != boards)).all()


def test_max_tile_and_empty_count():
    b = g.encode([[2, 0, 0, 0], [0, 2048, 0, 0], [0, 0, 4, 0], [0, 0, 0, 0]])
    assert g.max_tile(np.array([b], dtype=np.uint64))[0] == 11
    assert g.count_empty(np.array([b], dtype=np.uint64))[0] == 13
