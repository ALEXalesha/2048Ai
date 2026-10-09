"""Уровни ИИ и загрузка моделей.

- новичок: ранний снимок сети, без поиска;
- средний: обученная сеть, без поиска;
- сильный: обученная сеть и expectimax.

Модели лежат в game/models. Если модель больше 50 МБ, она идёт отдельным файлом выпуска
(spec §6): тогда её ищут ещё и рядом с программой - рядом с exe или в папке данных.
"""
from __future__ import annotations

import json
import os
import sys
import threading

import numpy as np

from game import paths
from game.ai import expectimax
from game.ai.ntuple import NTupleNet

LEVELS = ("novice", "medium", "strong")
MAIN_MODEL = "net.npz"
NOVICE_MODEL = "novice.npz"
PASSPORT = "passport.json"

# Настройки уровней. depth - глубина поиска в своих ходах; adaptive - на тесном поле (пустых
# клеток не больше deep_below) смотреть на ход дальше: там ветвей мало, а опасность близко.
# Замер на 100 партиях: глубина 2 - 8192 в 38 %, с углублением - в 63 %, но в 19 раз
# дольше. Те же числа - в паспорте.
LEVEL_SETTINGS = {
    "novice": {"model": NOVICE_MODEL, "depth": 1, "adaptive": False},
    "medium": {"model": MAIN_MODEL, "depth": 1, "adaptive": False},
    "strong": {"model": MAIN_MODEL, "depth": 2, "adaptive": True, "deep_below": 4},
}


def search_dirs(extra=()):
    dirs = [paths.models_dir()]
    if paths.frozen():
        dirs.append(os.path.dirname(os.path.abspath(sys.executable)))
    dirs.extend(extra)
    return dirs


def find_model(name, extra=()):
    """Путь к файлу модели или None."""
    for d in search_dirs(extra):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def load_passport():
    p = find_model(PASSPORT)
    if p is None:
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


class Brain:
    """ИИ для окна: держит загруженные сети и отвечает на вопрос «куда ходить»."""

    def __init__(self, extra_dirs=()):
        self.extra = tuple(extra_dirs)
        self._nets = {}
        # Взводится, когда окно закрывается: недогруженная модель бросается (LoadCancelled).
        self.stop_event = threading.Event()

    def available(self, level="strong"):
        return find_model(LEVEL_SETTINGS[level]["model"], self.extra) is not None

    def net(self, level):
        name = LEVEL_SETTINGS[level]["model"]
        if name not in self._nets:
            path = find_model(name, self.extra)
            if path is None:
                raise FileNotFoundError(name)
            self._nets[name] = NTupleNet.load(path, should_stop=self.stop_event.is_set)
        return self._nets[name]

    def values(self, board, level="strong", depth=None, time_limit=None):
        """Оценка четырёх ходов (UP, RIGHT, DOWN, LEFT) и глубина, на которой посчитано."""
        s = LEVEL_SETTINGS[level]
        d = depth or s["depth"]
        if s["adaptive"] and depth is None:
            d = int(expectimax.adaptive_depth(np.array([board], dtype=np.uint64), base=d,
                                              deep_below=s.get("deep_below", 4))[0])
        net = self.net(level)
        if time_limit is not None:
            res = expectimax.search(net, board, max_depth=d, time_limit=time_limit)
            return np.array(res.values), res.depth
        q = expectimax.action_values(net, np.array([board], dtype=np.uint64), d)[0]
        return q, d

    def move(self, board, level="strong"):
        q, _ = self.values(board, level)
        if not np.isfinite(q).any():
            return None
        return int(np.argmax(q))


def policy(brain, level):
    """Стратегия уровня для прогона партий пачкой (паспорт, тесты)."""
    s = LEVEL_SETTINGS[level]
    return expectimax.policy(brain.net(level), s["depth"], s["adaptive"], s.get("deep_below", 4))
