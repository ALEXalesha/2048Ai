"""Обучение TD на позициях после хода: учит, пишет журнал, снимки, продолжает с места."""
import json
import os

import numpy as np
import pytest

from game.ai import bots, ntuple
from game.ai.ntuple import NTupleNet
from game.core import game2048 as g
from game.train import td


def test_short_training_beats_random_play_by_far():
    net = NTupleNet(ntuple.TUPLES_6)
    trainer = td.Trainer(net, batch=256, alpha=0.1, seed=1)
    trainer.run(games=1500)
    assert trainer.games_done >= 1500
    trained = bots.summary(bots.play(bots.greedy_policy(net), 100, seed=500))
    rand = bots.summary(bots.play(bots.random_policy(0), 100, seed=500))
    corner = bots.summary(bots.play(bots.corner_policy(), 100, seed=500))
    # Замер: после 1500 партий ~16 тысяч очков против ~1 тысячи у случайных ходов и ~2.4
    # тысячи у «больших в угол». Сеть без оценки будущего (только очки хода) - ~3 тысячи.
    assert trained["avg_score"] > 8 * rand["avg_score"], (trained, rand)
    assert trained["avg_score"] > 4 * corner["avg_score"], (trained, corner)


def test_td_step_moves_value_towards_target():
    net = NTupleNet(ntuple.TUPLES_4)
    b = np.array([g.encode([[2, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]])], dtype=np.uint64)
    feats = net.features(b)
    before = net.value(b)[0]
    target = np.array([100.0])
    td.td_update(net, feats, target, lr=0.01)
    after = net.value(b)[0]
    assert before < after < 100.0


def test_terminal_afterstate_learns_zero():
    # Позиция, из которой после любой новой плитки ходов нет, должна стремиться к 0.
    net = NTupleNet(ntuple.TUPLES_4)
    net.w[:] = 5.0
    b = np.array([g.encode([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])], dtype=np.uint64)
    feats = net.features(b)
    for _ in range(200):
        td.td_update(net, feats, np.array([0.0]), lr=0.01)
    assert abs(net.value(b)[0]) < 1.0


def test_log_snapshots_and_resume(tmp_path):
    run = tmp_path / "run"
    net = NTupleNet(ntuple.TUPLES_4)
    tr = td.Trainer(net, batch=64, alpha=0.1, seed=3, run_dir=str(run), log_every=100,
                    snapshot_every=200, milestones=(150,))
    tr.run(games=400)
    assert (run / "latest.npy").exists() and (run / "state.json").exists()
    assert (run / "milestone-150.npy").exists()
    lines = [json.loads(x) for x in (run / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines and {"games", "avg_score", "reached_2048", "games_per_s", "elapsed_s"} <= set(lines[-1])
    assert (run / "log.txt").exists()
    state = json.loads((run / "state.json").read_text(encoding="utf-8"))
    assert state["games_done"] == tr.games_done >= 400

    back = td.Trainer.resume(str(run), ntuple.TUPLES_4)
    assert back.games_done == tr.games_done
    assert (back.net.w == np.load(run / "latest.npy")).all()
    done = back.games_done
    back.run(games=done + 100)
    assert back.games_done >= done + 100
    # Время прошлых запусков не теряется.
    assert back.seconds >= state["seconds"]


def test_stop_by_time_limit():
    net = NTupleNet(ntuple.TUPLES_4)
    tr = td.Trainer(net, batch=32, alpha=0.1, seed=4)
    tr.run(games=10 ** 9, seconds=1.5)
    assert 0 < tr.games_done < 10 ** 9


def test_threads_are_limited(monkeypatch):
    for var in td.THREAD_VARS:
        monkeypatch.delenv(var, raising=False)
    td.limit_threads(2)
    assert all(os.environ[v] == "2" for v in td.THREAD_VARS)


@pytest.mark.parametrize("lr", [0.0])
def test_zero_learning_rate_changes_nothing(lr):
    net = NTupleNet(ntuple.TUPLES_4)
    td.Trainer(net, batch=16, alpha=lr, seed=5).run(games=20)
    assert not net.w.any()
