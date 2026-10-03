"""Правила 2048 простым кодом, для поля любого размера.

Медленно, зато просто и прозрачно: это образец, с которым тесты сверяют быструю доску
4x4 (game2048), и это же правила окна для полей 3x3, 5x5 и 6x6. Заодно здесь считается,
какая плитка куда поехала, - для анимации.

Плитки хранятся числами (2, 4, 8...), пустая клетка - 0.
"""
from __future__ import annotations

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
DIRECTIONS = (UP, RIGHT, DOWN, LEFT)

# Предел быстрой доски: 4 бита на клетку, плитка не больше 2^15. Два 32768 не сливаются -
# так же и здесь, чтобы правила на всех полях были одни.
MAX_TILE = 32768
FOUR_CHANCE = 0.1


def slide_line(values):
    """Сдвиг одной линии к её началу. Возвращает (новая линия, очки, сдвинулось ли).

    Каждая плитка сливается за ход не больше одного раза: после слияния берётся
    следующая пара, а не результат слияния."""
    out, score, _ = slide_line_moves(values)
    return out, score, out != list(values)


def slide_line_moves(values):
    """То же, плюс куда поехала каждая плитка: список (откуда, куда, слилась ли)."""
    tiles = [(i, v) for i, v in enumerate(values) if v]
    out = []
    moves = []
    score = 0
    k = 0
    while k < len(tiles):
        i, v = tiles[k]
        if k + 1 < len(tiles) and tiles[k + 1][1] == v and v < MAX_TILE:
            dst = len(out)
            out.append(2 * v)
            score += 2 * v
            moves.append((i, dst, True))
            moves.append((tiles[k + 1][0], dst, True))
            k += 2
        else:
            moves.append((i, len(out), False))
            out.append(v)
            k += 1
    out += [0] * (len(values) - len(out))
    return out, score, moves


def lines(size, direction):
    """Координаты клеток по линиям, в порядке от края, к которому едут плитки."""
    rng = range(size)
    if direction == LEFT:
        return [[(r, c) for c in rng] for r in rng]
    if direction == RIGHT:
        return [[(r, c) for c in reversed(rng)] for r in rng]
    if direction == UP:
        return [[(r, c) for r in rng] for c in rng]
    if direction == DOWN:
        return [[(r, c) for r in reversed(rng)] for c in rng]
    raise ValueError(direction)


def slide(grid, direction):
    """Сдвиг всего поля: (новое поле, очки, сдвинулось ли, движения).

    Движения - список (откуда (r, c), куда (r, c), значение, слилась ли) для каждой плитки."""
    size = len(grid)
    new = [[0] * size for _ in range(size)]
    score = 0
    motions = []
    for line in lines(size, direction):
        values = [grid[r][c] for r, c in line]
        out, gained, moves = slide_line_moves(values)
        score += gained
        for (r, c), v in zip(line, out):
            new[r][c] = v
        for src, dst, merged in moves:
            motions.append((line[src], line[dst], values[src], merged))
    moved = new != [list(row) for row in grid]
    return new, score, moved, motions


def empty_cells(grid):
    return [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == 0]


def spawn(grid, rng):
    """Новая плитка в случайной пустой клетке: 2 (90 %) или 4 (10 %).

    Сначала выбирается клетка, потом значение - тот же порядок, что у быстрой доски, поэтому
    при одном зерне обе дают одни и те же плитки. Возвращает (r, c, значение) или None."""
    free = empty_cells(grid)
    if not free:
        return None
    k = int(rng.random() * len(free))
    value = 4 if rng.random() < FOUR_CHANCE else 2
    r, c = free[k]
    grid[r][c] = value
    return r, c, value


def can_move(grid):
    size = len(grid)
    for r in range(size):
        for c in range(size):
            v = grid[r][c]
            if v == 0:
                return True
            if c + 1 < size and grid[r][c + 1] == v and v < MAX_TILE:
                return True
            if r + 1 < size and grid[r + 1][c] == v and v < MAX_TILE:
                return True
    return False


class Game:
    """Одна партия на поле size x size. rng - numpy Generator, от него вся случайность."""

    def __init__(self, size, rng):
        self.size = size
        self.rng = rng
        self.grid = [[0] * size for _ in range(size)]
        self.score = 0
        self.moves = 0
        self.last_spawn = None
        spawn(self.grid, rng)
        self.last_spawn = spawn(self.grid, rng)

    def move(self, direction):
        """Сделать ход. Возвращает (сдвинулось ли, очки за ход, движения, новая плитка)."""
        new, gained, moved, motions = slide(self.grid, direction)
        if not moved:
            return False, 0, [], None
        self.grid = new
        self.score += gained
        self.moves += 1
        self.last_spawn = spawn(self.grid, self.rng)
        return True, gained, motions, self.last_spawn

    def legal(self, direction):
        return slide(self.grid, direction)[2]

    def legal_moves(self):
        return [d for d in DIRECTIONS if self.legal(d)]

    def over(self):
        return not can_move(self.grid)

    def max_tile(self):
        return max(max(row) for row in self.grid)

    def snapshot(self):
        """Всё состояние партии, включая генератор случайности: после отмены хода та же
        плитка появится там же - отменой не перебрать удачную."""
        return ([row[:] for row in self.grid], self.score, self.moves, self.last_spawn,
                self.rng.bit_generator.state)

    def restore(self, snap):
        grid, self.score, self.moves, self.last_spawn, state = snap
        self.grid = [row[:] for row in grid]
        self.rng.bit_generator.state = state
