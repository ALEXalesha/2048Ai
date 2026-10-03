"""Партия окна: правила generic на поле любого размера, отмена хода, связь с быстрой
доской для ИИ. Без Qt - проверяется обычными тестами."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from game.core import game2048 as g
from game.core import generic

UNDO_DEPTH = 100
TARGET = 2048


@dataclass
class MoveResult:
    moved: bool
    gained: int = 0
    motions: list = field(default_factory=list)   # (откуда, куда, значение, слилась ли)
    spawn: tuple | None = None                     # (r, c, значение)
    reached: int | None = None                     # впервые собрана 2048 (дальше - 4096...)
    over: bool = False


class Session:
    """Одна партия. seed задаёт всю случайность: две партии с одним зерном и одними
    ходами получают одни и те же плитки (режим «Соревнование»)."""

    def __init__(self, size=4, seed=None, undo=False):
        self.size = size
        self.seed = int(seed) if seed is not None else int(np.random.default_rng().integers(1 << 62))
        self.game = generic.Game(size, np.random.default_rng(self.seed))
        self.undo_allowed = undo
        self.history = []
        self.reached = 0
        self.finished_recorded = False

    @property
    def grid(self):
        return self.game.grid

    @property
    def score(self):
        return self.game.score

    @property
    def moves(self):
        return self.game.moves

    def max_tile(self):
        return self.game.max_tile()

    def over(self):
        return self.game.over()

    def board64(self):
        """Поле для ИИ (только 4x4) или None."""
        if self.size != 4:
            return None
        return g.encode(self.grid)

    def move(self, direction):
        snap = self.game.snapshot() + (self.reached,)
        moved, gained, motions, spawn = self.game.move(direction)
        if not moved:
            return MoveResult(False)
        if self.undo_allowed:
            self.history.append(snap)
            del self.history[:-UNDO_DEPTH]
        reached = None
        top = self.max_tile()
        if top >= TARGET and top > self.reached and _is_milestone(top):
            # Сообщаем о 2048 и о каждой следующей «круглой» плитке по одному разу.
            reached = top
        if top >= TARGET:
            self.reached = max(self.reached, top)
        return MoveResult(True, gained, motions, spawn, reached, self.over())

    def can_undo(self):
        return self.undo_allowed and bool(self.history)

    def undo(self):
        if not self.can_undo():
            return False
        snap = self.history.pop()
        self.game.restore(snap[:5])
        self.reached = snap[5]
        return True


def _is_milestone(tile):
    return tile >= TARGET and (tile & (tile - 1)) == 0
