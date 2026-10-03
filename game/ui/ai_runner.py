"""ИИ в отдельном потоке: окно не замирает, пока ИИ думает или грузит модель.

Каждый запрос получает номер. Ответ на устаревший запрос (сменили режим, начали новую
партию) приходит, но отбрасывается. stop() останавливает поток за 2 секунды.
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot

from game.ai import levels
from game.core import generic

CORNER = [generic.DOWN, generic.LEFT, generic.RIGHT, generic.UP]


def corner_move(grid):
    """«Большие в угол» на поле любого размера: первый разрешённый из вниз, влево, вправо, вверх."""
    for d in CORNER:
        if generic.slide(grid, d)[2]:
            return d
    return None


class Worker(QObject):
    done = Signal(int, object)

    def __init__(self, brain):
        super().__init__()
        self.brain = brain

    @Slot(int, object)
    def compute(self, token, job):
        try:
            result = self._compute(job)
        except Exception as exc:  # noqa: BLE001 - ошибка уходит в окно, поток живёт
            result = {"error": repr(exc), "move": None, "values": [None] * 4}
        self.done.emit(token, result)

    def _compute(self, job):
        grid = job["grid"]
        if len(grid) != 4 or not self.brain.available(job.get("level", "strong")):
            move = corner_move(grid)
            values = [None] * 4
            if move is not None:
                values = [1.0 if d == move else (0.0 if generic.slide(grid, d)[2] else -math.inf)
                          for d in range(4)]
            return {"move": move, "values": values, "depth": 0, "fallback": True}
        from game.core import game2048 as g
        board = np.uint64(g.encode(grid))
        level = job.get("level", "strong")
        q, depth = self.brain.values(board, level, depth=job.get("depth"),
                                     time_limit=job.get("time_limit"))
        q = [float(x) for x in q]
        finite = [x for x in q if math.isfinite(x)]
        move = int(np.argmax(q)) if finite else None
        return {"move": move, "values": q, "depth": depth, "fallback": False}


class AiRunner(QObject):
    request = Signal(int, object)
    answered = Signal(object, object)       # (запрос, ответ)

    def __init__(self, brain=None, parent=None):
        super().__init__(parent)
        self.brain = brain or levels.Brain()
        self.token = 0
        self.pending = {}
        self.thread = QThread()
        self.thread.setObjectName("ai2048-worker")
        self.worker = Worker(self.brain)
        self.worker.moveToThread(self.thread)
        self.request.connect(self.worker.compute)
        self.worker.done.connect(self._on_done)
        self.thread.start()

    def ask(self, job):
        self.token += 1
        self.pending = {self.token: job}
        self.request.emit(self.token, job)
        return self.token

    def busy(self):
        return bool(self.pending)

    def cancel(self):
        self.token += 1
        self.pending = {}

    def _on_done(self, token, result):
        job = self.pending.pop(token, None)
        if job is None:
            return
        self.answered.emit(job, result)

    def stop(self, timeout_ms=2000):
        self.cancel()
        self.thread.quit()
        return self.thread.wait(timeout_ms)


def next_tile_chance(passport, max_tile, level="strong"):
    """Вероятность дойти до следующей плитки при нынешней наибольшей - по паспорту:
    доля партий, где была следующая, среди партий, где была нынешняя. (плитка, доля или None)."""
    nxt = int(max_tile) * 2 if max_tile else 4
    res = (passport.get("results") or {}).get(level)
    if not res or "max_tile_counts" not in res:
        return nxt, None
    counts = {int(k): v for k, v in res["max_tile_counts"].items()}
    have = sum(v for k, v in counts.items() if k >= max_tile)
    reach = sum(v for k, v in counts.items() if k >= nxt)
    if have == 0:
        return nxt, None
    return nxt, reach / have


__all__ = ["AiRunner", "corner_move", "next_tile_chance"]
