"""Партия окна (без Qt): отмена хода, одно зерно для двух полей, рекорды."""
import numpy as np

from game.core import game2048 as g
from game.core.records import Records
from game.core.session import Session


def test_undo_restores_board_score_and_the_same_next_tile():
    s = Session(4, seed=11, undo=True)
    start = [row[:] for row in s.grid]
    moved = next(d for d in range(4) if s.move(d).moved)
    after_first = [row[:] for row in s.grid]
    assert s.can_undo()
    s.undo()
    assert s.grid == start and s.score == 0 and s.moves == 0
    s.move(moved)
    assert s.grid == after_first          # отмена не даёт перебрать новую плитку


def test_undo_is_off_when_not_allowed():
    s = Session(4, seed=1, undo=False)
    d = next(d for d in range(4) if s.game.legal(d))
    s.move(d)
    assert not s.can_undo()
    assert s.undo() is False


def test_same_seed_gives_the_same_tiles_for_the_same_moves():
    a, b = Session(4, seed=5), Session(4, seed=5)
    assert a.grid == b.grid
    rng = np.random.default_rng(0)
    for _ in range(50):
        legal = a.game.legal_moves()
        if not legal:
            break
        d = int(rng.choice(legal))
        a.move(d)
        b.move(d)
        assert a.grid == b.grid and a.score == b.score


def test_board64_matches_the_grid():
    s = Session(4, seed=3)
    assert g.decode(s.board64()).tolist() == s.grid
    assert Session(5, seed=3).board64() is None


def test_reaching_2048_is_reported_once_and_game_continues():
    s = Session(4, seed=0)
    s.game.grid = [[1024, 1024, 0, 0], [0] * 4, [0] * 4, [0] * 4]
    r = s.move(3)       # влево
    assert r.reached == 2048 and r.moved
    assert not s.over()
    s.game.grid = [[2048, 0, 0, 2], [0] * 4, [0] * 4, [0] * 4]
    r = s.move(3)
    assert r.reached is None


def test_records_keep_best_and_totals(tmp_path):
    path = tmp_path / "records.json"
    rec = Records(str(path))
    rec.add("me", 4, score=1000, max_tile=128)
    rec.add("me", 4, score=25000, max_tile=2048)
    rec.add("ai", 4, score=60000, max_tile=4096)
    back = Records(str(path))
    me = back.get("me", 4)
    assert me["games"] == 2 and me["best_score"] == 25000 and me["best_tile"] == 2048
    assert me["total_score"] == 26000 and me["reached_2048"] == 1
    assert back.get("ai", 4)["best_tile"] == 4096
    assert back.get("me", 5)["games"] == 0


def test_records_survive_garbage_file(tmp_path):
    path = tmp_path / "records.json"
    path.write_text("{oops", encoding="utf-8")
    rec = Records(str(path))
    assert rec.get("me", 4)["games"] == 0
    path.write_text('{"me/4": {"games": "много"}, "ai/4": {"games": 1, "best_score": 5, '
                    '"best_tile": 4, "total_score": 5, "reached_2048": 0}}', encoding="utf-8")
    rec = Records(str(path))
    assert rec.get("me", 4)["games"] == 0
    assert rec.get("ai", 4)["games"] == 1
