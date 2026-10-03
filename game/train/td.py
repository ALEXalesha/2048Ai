"""Обучение n-tuple сети методом временных разностей (TD(0)) на позициях после хода.

Сеть играет сама с собой - точнее, со случайностью. В каждой позиции она пробует все
четыре хода и берёт тот, где «очки за ход + оценка позиции после хода» больше. Потом
появляется новая плитка, и в следующей позиции сеть снова выбирает ход. Оценка прошлой
позиции после хода подтягивается к тому, что выяснилось: очки следующего хода плюс оценка
следующей позиции после хода (а если ходов больше нет - к нулю). Szubert и Jaśkowski,
2014: такой «afterstate»-вариант учится лучше всего.

Партий в работе сразу много (пачка), шаг идёт во всех разом: numpy считает тысячи досок за
одну операцию. Закончилась партия - на её место встаёт новая.

Журнал (log.jsonl для программ и log.txt для глаз), снимки весов и продолжение с места -
в папке запуска. Потоки ограничены: обучение делит ПК с другими задачами.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque

import numpy as np

from game.ai.ntuple import TUPLES_6, NTupleNet
from game.core import game2048 as g

THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def limit_threads(n):
    """Потоки библиотек numpy. Звать до импорта numpy, чтобы подействовало наверняка;
    выборки из таблиц numpy и так считает в одном потоке."""
    for var in THREAD_VARS:
        os.environ[var] = str(n)


def lower_priority():
    """Пониженный приоритет процесса в Windows: ПК остаётся отзывчивым."""
    if os.name != "nt":
        return False
    try:
        import ctypes
        below_normal = 0x00004000
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        return bool(ctypes.windll.kernel32.SetPriorityClass(handle, below_normal))
    except Exception:  # noqa: BLE001 - приоритет не обязателен
        return False


def td_update(net, feats, target, lr):
    """Подтянуть оценки позиций (по их номерам весов) к цели. Возвращает ошибку."""
    delta = np.asarray(target, dtype=np.float64) - net.value_of_features(feats)
    net.update(feats, delta, lr)
    return delta


def _save_atomic_npy(path, array):
    tmp = path + ".tmp.npy"
    np.save(tmp, array)
    os.replace(tmp, path)


def _write_atomic(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


class Trainer:
    """Обучение пачкой партий. alpha - шаг на группу клеток: каждый из 8 симметричных
    весов группы получает alpha/8 от ошибки (как в TDL2048 у moporgic)."""

    def __init__(self, net, batch=1024, alpha=0.1, seed=0, run_dir=None, log_every=1000,
                 snapshot_every=10000, milestones=(), window=1000, echo=False):
        self.net = net
        self.batch = int(batch)
        self.alpha = float(alpha)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.run_dir = run_dir
        self.log_every = int(log_every)
        self.snapshot_every = int(snapshot_every)
        self.milestones = sorted(int(m) for m in milestones)
        self.echo = echo
        self.games_done = 0
        self.moves_done = 0
        self.seconds = 0.0
        self.best_score = 0
        self.recent = deque(maxlen=int(window))
        if run_dir:
            os.makedirs(run_dir, exist_ok=True)

    @property
    def lr(self):
        return self.alpha * len(self.net.tuples) / self.net.n_features

    # --- обучение -----------------------------------------------------------------------

    def run(self, games, seconds=None, should_stop=None):
        """Учиться, пока всего сыграно меньше games партий (и не вышло seconds секунд)."""
        net, rng, n = self.net, self.rng, self.batch
        boards, scores, _ = g.new_games(n, rng)
        moves = np.zeros(n, dtype=np.int64)
        prev = np.zeros((n, net.n_features), dtype=np.int64)
        has_prev = np.zeros(n, dtype=bool)
        idx = np.arange(n)
        started = time.time()
        base_seconds = self.seconds
        next_log = (self.games_done // self.log_every + 1) * self.log_every
        next_snap = (self.games_done // self.snapshot_every + 1) * self.snapshot_every
        pending = [m for m in self.milestones if m > self.games_done]
        start_games, start_moves = self.games_done, self.moves_done

        while self.games_done < games:
            if seconds is not None and time.time() - started >= seconds:
                break
            if should_stop is not None and should_stop():
                break
            after, reward, legal = g.all_moves(boards)
            feats = net.features(after.ravel()).reshape(n, 4, -1)
            q = reward + net.w[feats].sum(axis=2, dtype=np.float64)
            q[~legal] = -np.inf
            a = q.argmax(axis=1)
            dead = ~legal.any(axis=1)
            target = np.where(dead, 0.0, q[idx, a])
            if has_prev.any():
                td_update(net, prev[has_prev], target[has_prev], self.lr)

            alive = ~dead
            prev = feats[idx, a]
            has_prev = alive.copy()
            scores = scores + np.where(alive, reward[idx, a], 0)
            moves += alive
            nb = after[idx, a]
            nb[alive] = g.spawn(nb[alive], rng)
            boards = nb

            if dead.any():
                d = np.flatnonzero(dead)
                tops = g.max_tile(boards[d])
                for k, top in zip(d, tops):
                    self.recent.append((int(scores[k]), int(top), int(moves[k])))
                    self.best_score = max(self.best_score, int(scores[k]))
                self.games_done += len(d)
                self.moves_done += int(moves[d].sum())
                fresh, _, _ = g.new_games(len(d), rng)
                boards[d] = fresh
                scores[d] = 0
                moves[d] = 0
                has_prev[d] = False

                self.seconds = base_seconds + time.time() - started
                if self.games_done >= next_log:
                    self._log(start_games, start_moves, time.time() - started)
                    next_log = (self.games_done // self.log_every + 1) * self.log_every
                while pending and self.games_done >= pending[0]:
                    self._snapshot(f"milestone-{pending.pop(0)}")
                if self.run_dir and self.games_done >= next_snap:
                    self.save()
                    next_snap = (self.games_done // self.snapshot_every + 1) * self.snapshot_every

        self.seconds = base_seconds + time.time() - started
        if self.run_dir:
            self.save()
        return self.stats()

    # --- журнал и снимки ----------------------------------------------------------------

    def stats(self):
        rec = np.array(self.recent, dtype=np.int64).reshape(-1, 3)
        out = {"games": self.games_done, "moves": self.moves_done,
               "elapsed_s": round(self.seconds, 1), "best_score": self.best_score,
               "alpha": self.alpha, "batch": self.batch}
        if len(rec):
            out["avg_score"] = round(float(rec[:, 0].mean()), 1)
            out["avg_moves"] = round(float(rec[:, 2].mean()), 1)
            for tile in (2048, 4096, 8192, 16384):
                e = tile.bit_length() - 1
                out[f"reached_{tile}"] = round(float((rec[:, 1] >= e).mean()), 4)
        else:
            out["avg_score"] = 0.0
            out["avg_moves"] = 0.0
            for tile in (2048, 4096, 8192, 16384):
                out[f"reached_{tile}"] = 0.0
        return out

    def _log(self, start_games, start_moves, run_seconds):
        s = self.stats()
        run_seconds = max(run_seconds, 1e-9)
        s["games_per_s"] = round((self.games_done - start_games) / run_seconds, 2)
        s["moves_per_s"] = round((self.moves_done - start_moves) / run_seconds, 1)
        s["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        line = (f"{s['time']}  партий {s['games']:>8}  средний счёт {s['avg_score']:>9.0f}  "
                f"2048 {100 * s['reached_2048']:5.1f}%  4096 {100 * s['reached_4096']:5.1f}%  "
                f"8192 {100 * s['reached_8192']:5.1f}%  лучший {s['best_score']:>7}  "
                f"{s['games_per_s']:.1f} партий/с  {s['moves_per_s']:.0f} ходов/с  "
                f"прошло {s['elapsed_s'] / 3600:.2f} ч")
        if self.echo:
            print(line, flush=True)
        if self.run_dir:
            with open(os.path.join(self.run_dir, "log.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
            with open(os.path.join(self.run_dir, "log.txt"), "a", encoding="utf-8") as f:
                f.write(line + "\n")
        return s

    def _snapshot(self, name):
        if self.run_dir:
            _save_atomic_npy(os.path.join(self.run_dir, f"{name}.npy"), self.net.w)

    def save(self):
        """Снимок для продолжения: веса и состояние (сколько сыграно, время, генератор)."""
        if not self.run_dir:
            return
        _save_atomic_npy(os.path.join(self.run_dir, "latest.npy"), self.net.w)
        state = {
            "games_done": self.games_done, "moves_done": self.moves_done,
            "seconds": self.seconds, "best_score": self.best_score,
            "alpha": self.alpha, "batch": self.batch, "seed": self.seed,
            "log_every": self.log_every, "snapshot_every": self.snapshot_every,
            "milestones": self.milestones, "window": self.recent.maxlen,
            "tuples": [list(t) for t in self.net.tuples],
            "recent": list(self.recent),
            "rng": self.rng.bit_generator.state,
            "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _write_atomic(os.path.join(self.run_dir, "state.json"), json.dumps(state, ensure_ascii=False))

    @classmethod
    def resume(cls, run_dir, tuples=None, **overrides):
        """Продолжить с последнего снимка. overrides - например alpha или batch."""
        with open(os.path.join(run_dir, "state.json"), encoding="utf-8") as f:
            state = json.load(f)
        tuples = tuples or [tuple(t) for t in state["tuples"]]
        net = NTupleNet.load_raw(os.path.join(run_dir, "latest.npy"), tuples)
        params = {k: state[k] for k in ("batch", "alpha", "seed", "log_every", "snapshot_every",
                                        "milestones", "window")}
        params.update(overrides)
        tr = cls(net, run_dir=run_dir, **params)
        tr.games_done = state["games_done"]
        tr.moves_done = state["moves_done"]
        tr.seconds = state["seconds"]
        tr.best_score = state["best_score"]
        tr.recent.extend(tuple(x) for x in state["recent"])
        tr.rng.bit_generator.state = state["rng"]
        return tr


def new_trainer(run_dir=None, tuples=TUPLES_6, **kw):
    return Trainer(NTupleNet(tuples), run_dir=run_dir, **kw)
