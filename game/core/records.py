"""Рекорды: партии игрока и партии ИИ по размерам поля. Битый файл - пустые рекорды,
битая запись пропускается, остальные остаются."""
from __future__ import annotations

from aikit.storage import is_int, load_json, save_json

FIELDS = ("games", "best_score", "best_tile", "total_score", "reached_2048")
WHO = ("me", "ai")


def empty():
    return {f: 0 for f in FIELDS}


class Records:
    def __init__(self, path=None):
        self.path = path
        self.data = {}
        raw = load_json(path) if path else None
        if isinstance(raw, dict):
            for k, e in raw.items():
                if isinstance(e, dict) and all(is_int(e.get(f)) and e[f] >= 0 for f in FIELDS):
                    self.data[k] = {f: e[f] for f in FIELDS}

    @staticmethod
    def key(who, size):
        return f"{who}/{size}"

    def get(self, who, size):
        return dict(self.data.get(self.key(who, size), empty()))

    def add(self, who, size, score, max_tile):
        e = self.get(who, size)
        e["games"] += 1
        e["best_score"] = max(e["best_score"], int(score))
        e["best_tile"] = max(e["best_tile"], int(max_tile))
        e["total_score"] += int(score)
        e["reached_2048"] += int(max_tile >= 2048)
        self.data[self.key(who, size)] = e
        if self.path:
            save_json(self.path, self.data)
        return e

    def best_score(self, size):
        return max(self.get(w, size)["best_score"] for w in WHO)
