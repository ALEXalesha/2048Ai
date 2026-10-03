"""Быстрые правила 2048 для поля 4x4 - на них учится и думает ИИ.

Поле - одно 64-битное число: 16 клеток по 4 бита. В клетке не сама плитка, а её степень
двойки: 0 - пусто, 1 - плитка 2, 11 - 2048, 15 - 32768 (больше в 4 бита не влезает, два
32768 не сливаются). Клетка (r, c) лежит в битах 4*(4r + c): строка r - это 16 бит
начиная с 16r, клетка c - младшие биты строки.

Сдвиг строки берётся из готовой таблицы на все 65536 строк, вверх и вниз - через
транспонирование. Все функции работают с массивами numpy uint64: тысячи партий
считаются разом, без цикла Python по партиям.
"""
from __future__ import annotations

import numpy as np

from game.core import generic

UP, RIGHT, DOWN, LEFT = generic.UP, generic.RIGHT, generic.DOWN, generic.LEFT
DIRECTIONS = generic.DIRECTIONS
MAX_EXP = 15
FOUR_CHANCE = generic.FOUR_CHANCE

U = np.uint64
_ROW = U(0xFFFF)
_NIB = U(0xF)
_SHIFTS16 = np.arange(16, dtype=np.uint64) * U(4)


def _slide_exps(cells):
    """Сдвиг влево строки из степеней. Возвращает (новые степени, очки)."""
    tiles = [e for e in cells if e]
    out = []
    score = 0
    k = 0
    while k < len(tiles):
        e = tiles[k]
        if k + 1 < len(tiles) and tiles[k + 1] == e and e < MAX_EXP:
            out.append(e + 1)
            score += 1 << (e + 1)
            k += 2
        else:
            out.append(e)
            k += 1
    out += [0] * (4 - len(out))
    return out, score


def _pack(cells):
    return cells[0] | (cells[1] << 4) | (cells[2] << 8) | (cells[3] << 12)


def _build_tables():
    left = np.zeros(1 << 16, dtype=np.uint64)
    right = np.zeros(1 << 16, dtype=np.uint64)
    score = np.zeros(1 << 16, dtype=np.int64)
    score_right = np.zeros(1 << 16, dtype=np.int64)
    for r in range(1 << 16):
        cells = [(r >> (4 * c)) & 0xF for c in range(4)]
        out, s = _slide_exps(cells)
        left[r] = _pack(out)
        score[r] = s
        rev, s = _slide_exps(cells[::-1])
        right[r] = _pack(rev[::-1])
        score_right[r] = s
    return left, right, score, score_right


ROW_LEFT, ROW_RIGHT, ROW_SCORE, ROW_SCORE_RIGHT = _build_tables()


# --- кодирование ----------------------------------------------------------------------

def encode_exps(grid) -> int:
    """Поле 4x4 из степеней -> число."""
    b = 0
    for r in range(4):
        for c in range(4):
            b |= int(grid[r][c]) << (4 * (4 * r + c))
    return b


def encode(grid) -> int:
    """Поле 4x4 из плиток (0, 2, 4, ...) -> число."""
    return encode_exps([[0 if v == 0 else int(v).bit_length() - 1 for v in row] for row in grid])


def exps(board) -> np.ndarray:
    """Число -> поле 4x4 из степеней."""
    b = U(board)
    return ((b >> _SHIFTS16) & _NIB).astype(np.int64).reshape(4, 4)


def decode(board) -> np.ndarray:
    """Число -> поле 4x4 из плиток."""
    e = exps(board)
    return np.where(e == 0, 0, 1 << e)


def cells(boards) -> np.ndarray:
    """Степени всех клеток: массив (N, 16)."""
    return ((boards[:, None] >> _SHIFTS16) & _NIB).astype(np.uint8)


# --- сдвиги ---------------------------------------------------------------------------

def transpose(b):
    """Строки становятся столбцами: клетка (r, c) переезжает в (c, r)."""
    a1 = b & U(0xF0F00F0FF0F00F0F)
    a2 = b & U(0x0000F0F00000F0F0)
    a3 = b & U(0x0F0F00000F0F0000)
    a = a1 | (a2 << U(12)) | (a3 >> U(12))
    b1 = a & U(0xFF00FF0000FF00FF)
    b2 = a & U(0x00FF00FF00000000)
    b3 = a & U(0x00000000FF00FF00)
    return b1 | (b2 >> U(24)) | (b3 << U(24))


def _rows(b, table, score):
    out = np.zeros_like(b)
    gained = np.zeros(b.shape, dtype=np.int64)
    for r in range(4):
        shift = U(16 * r)
        idx = ((b >> shift) & _ROW).astype(np.intp)
        out |= table[idx] << shift
        gained += score[idx]
    return out, gained


def _left(b):
    return _rows(b, ROW_LEFT, ROW_SCORE)


def _right(b):
    return _rows(b, ROW_RIGHT, ROW_SCORE_RIGHT)


def move(boards, direction):
    """Сдвиг всех полей в одну сторону: (новые поля, очки за ход). Поля, где ничего не
    сдвинулось, возвращаются как есть - ход по правилам запрещён."""
    boards = np.asarray(boards, dtype=np.uint64)
    if direction == LEFT:
        return _left(boards)
    if direction == RIGHT:
        return _right(boards)
    t = transpose(boards)
    if direction == UP:
        out, gained = _left(t)
    elif direction == DOWN:
        out, gained = _right(t)
    else:
        raise ValueError(direction)
    return transpose(out), gained


def all_moves(boards):
    """Все четыре хода разом: (поля после хода (N, 4), очки (N, 4), можно ли (N, 4)).
    Порядок ходов - UP, RIGHT, DOWN, LEFT."""
    boards = np.asarray(boards, dtype=np.uint64)
    t = transpose(boards)
    up, s_up = _left(t)
    down, s_down = _right(t)
    left, s_left = _left(boards)
    right, s_right = _right(boards)
    after = np.stack([transpose(up), right, transpose(down), left], axis=1)
    reward = np.stack([s_up, s_right, s_down, s_left], axis=1)
    legal = after != boards[:, None]
    return after, reward, legal


def game_over(boards):
    _, _, legal = all_moves(boards)
    return ~legal.any(axis=1)


def count_empty(boards):
    return (cells(np.asarray(boards, dtype=np.uint64)) == 0).sum(axis=1)


def max_tile(boards):
    """Наибольшая плитка каждого поля - степенью двойки (11 - это 2048)."""
    return cells(np.asarray(boards, dtype=np.uint64)).max(axis=1)


# --- новая плитка ---------------------------------------------------------------------

def spawn(boards, rng):
    """Новая плитка в случайной пустой клетке каждого поля: 2 (90 %) или 4 (10 %).
    Поля без пустых клеток не меняются. Порядок случайных чисел - как у generic.spawn:
    сначала клетки всех полей, потом значения."""
    boards = np.asarray(boards, dtype=np.uint64)
    n = boards.shape[0]
    u_pos = rng.random(n)
    u_val = rng.random(n)
    empty = cells(boards) == 0
    cnt = empty.sum(axis=1)
    k = np.floor(u_pos * cnt).astype(np.int64)
    cum = np.cumsum(empty, axis=1)
    pos = (cum <= k[:, None]).sum(axis=1)
    val = np.where(u_val < FOUR_CHANCE, 2, 1).astype(np.uint64)
    add = np.where(cnt > 0, val << (pos.astype(np.uint64) * U(4)), U(0))
    return boards | add


def new_games(n, rng):
    """n новых партий: (поля, счёт, жива ли). На поле две плитки."""
    boards = np.zeros(n, dtype=np.uint64)
    boards = spawn(boards, rng)
    boards = spawn(boards, rng)
    return boards, np.zeros(n, dtype=np.int64), np.ones(n, dtype=bool)


def play_step(boards, scores, alive, moves, rng):
    """Один ход во всех живых партиях. Запрещённый ход ничего не меняет (и плитка не
    появляется), законченные партии стоят. Возвращает (поля, счёт, жива ли)."""
    boards = np.asarray(boards, dtype=np.uint64)
    after, reward, legal = all_moves(boards)
    idx = np.arange(boards.shape[0])
    moves = np.asarray(moves)
    ok = legal[idx, moves] & alive
    new = np.where(ok, after[idx, moves], boards)
    # Плитка появляется только там, где ход был: случайные числа тратятся лишь на эти поля
    # (как у generic.Game, иначе при одном зерне доски разошлись бы).
    new[ok] = spawn(new[ok], rng)
    scores = np.where(ok, scores + reward[idx, moves], scores)
    alive = alive & ~game_over(new)
    return new, scores, alive
