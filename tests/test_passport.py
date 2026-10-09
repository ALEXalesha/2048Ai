"""Паспорт модели: цифры на месте, и малая выборка пересчитывается с теми же зёрнами
до очка. Сильнее уровень - больше счёт; все уровни сильнее простых ботов."""
import os

import numpy as np
import pytest

from game.ai import bots, levels
from game.ai.levels import Brain
from game.train import passport as pp

PASSPORT = levels.load_passport()
NAMES = ("random", "corner", "novice", "medium", "strong")


def test_models_and_passport_are_found():
    assert levels.find_model(levels.MAIN_MODEL)
    assert levels.find_model(levels.NOVICE_MODEL)
    assert PASSPORT, "нет game/models/passport.json"


def test_models_fit_in_the_repository_or_follow_the_release_asset_rule():
    # spec §6: больше 50 МБ - модель идёт файлом выпуска, а не в git.
    for name in (levels.MAIN_MODEL, levels.NOVICE_MODEL):
        path = levels.find_model(name)
        assert os.path.getsize(path) <= 50 * 1024 * 1024, name


def test_passport_has_every_level_and_bot_on_1000_games():
    assert PASSPORT["eval"]["games"] == 1000
    for name in NAMES:
        r = PASSPORT["results"][name]
        assert r["games"] == 1000
        assert set(r["reached"]) >= {"2048", "4096", "8192", "16384"}
        assert r["best_score"] >= r["avg_score"] > 0
        assert sum(r["max_tile_counts"].values()) == 1000
    t = PASSPORT["training"]
    assert t["games"] > 0 and t["hours"] > 0
    assert PASSPORT["level_settings"]["strong"]["depth"] >= 2


def test_stronger_level_scores_more():
    r = PASSPORT["results"]
    avg = {k: r[k]["avg_score"] for k in NAMES}
    assert avg["random"] < avg["corner"] < avg["novice"] < avg["medium"] < avg["strong"]
    assert r["strong"]["reached"]["2048"] >= r["medium"]["reached"]["2048"]


@pytest.mark.parametrize("name", NAMES)
def test_small_sample_is_recomputed_with_the_same_seeds(name):
    r = PASSPORT["results"][name]
    want = r["first_scores"]
    n = len(want)
    assert n == pp.SAMPLE[name]
    brain = Brain()
    policy = pp.players(brain)[name]()
    got = bots.play(policy, n, r["seed"])["score"]
    assert [int(x) for x in got] == want


def test_play_does_not_depend_on_the_batch_size():
    corner = bots.corner_policy()
    a = bots.play(corner, 12, seed=5, batch=12)["score"]
    b = bots.play(corner, 12, seed=5, batch=5)["score"]
    assert (np.asarray(a) == np.asarray(b)).all()


def test_random_bot_plays_legal_moves_and_does_not_depend_on_the_batch():
    rnd = bots.random_policy(3)
    a = bots.play(rnd, 10, seed=9, batch=10)
    b = bots.play(rnd, 10, seed=9, batch=3)
    assert (a["score"] == b["score"]).all()
    s = bots.summary(bots.play(rnd, 200, seed=1))
    assert 600 < s["avg_score"] < 2500          # случайные ходы: около тысячи очков


def test_parallel_passport_gives_the_same_games_as_one_process():
    one = bots.play(bots.corner_policy(), 6, pp.SEED)
    many = pp.play_parallel("corner", 6, processes=2, echo=None)
    for k in ("score", "max_exp", "moves"):
        assert (np.asarray(one[k]) == np.asarray(many[k])).all()


def test_passport_resumes_chunk_by_chunk(tmp_path):
    """Прерванный прогон не теряет готовые куски: они лежат файлами и берутся снова."""
    parts = tmp_path / "parts"
    first = pp.play_parallel("corner", 6, processes=1, echo=None, parts_dir=str(parts),
                             chunk=2, limit_chunks=1)
    assert first is None                                   # прогон не доделан
    files = sorted(parts.glob("corner-*.npz"))
    assert len(files) == 1
    stamp = files[0].stat().st_mtime_ns
    done = pp.play_parallel("corner", 6, processes=2, echo=None, parts_dir=str(parts), chunk=2)
    assert files[0].stat().st_mtime_ns == stamp            # готовый кусок не пересчитан
    assert len(list(parts.glob("corner-*.npz"))) == 3
    one = bots.play(bots.corner_policy(), 6, pp.SEED)
    for k in ("score", "max_exp", "moves"):
        assert (np.asarray(one[k]) == np.asarray(done[k])).all()


def test_parts_of_another_model_are_not_reused(tmp_path, monkeypatch):
    parts = tmp_path / "parts"
    pp.play_parallel("corner", 2, processes=1, echo=None, parts_dir=str(parts), chunk=2)
    monkeypatch.setattr(pp, "fingerprint", lambda name: "другая-модель")
    pp.play_parallel("corner", 2, processes=1, echo=None, parts_dir=str(parts), chunk=2)
    assert len(list(parts.glob("corner-*.npz"))) == 2
