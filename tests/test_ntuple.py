"""N-tuple сеть: симметрии, номера весов, оценка, файлы."""
import numpy as np
import pytest

from game.ai import ntuple
from game.ai.ntuple import NTupleNet
from game.core import game2048 as g


def random_boards(n, seed=0):
    rng = np.random.default_rng(seed)
    grids = rng.integers(0, 16, size=(n, 4, 4))
    return grids, np.array([g.encode_exps(x) for x in grids], dtype=np.uint64)


def test_eight_symmetries_are_rotations_and_reflections():
    grids, boards = random_boards(50)
    syms = ntuple.symmetries(boards)
    assert syms.shape == (8, 50)
    for k, x in enumerate(grids):
        want = []
        for m in (x, x.T):
            for r in range(4):
                want.append(np.rot90(m, r))
        want = {w.tobytes() for w in want}
        got = {g.exps(syms[s, k]).astype(x.dtype).tobytes() for s in range(8)}
        assert got == want


def test_flips():
    grid = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]
    b = np.array([g.encode_exps(grid)], dtype=np.uint64)
    assert g.exps(ntuple.flip_h(b)[0]).tolist() == [r[::-1] for r in grid]
    assert g.exps(ntuple.flip_v(b)[0]).tolist() == grid[::-1]


def slow_index(grid, cells):
    flat = np.asarray(grid).ravel()
    return sum(int(flat[c]) << (4 * j) for j, c in enumerate(cells))


@pytest.mark.parametrize("tuples", [ntuple.TUPLES_6, ntuple.TUPLES_4, ((0, 5, 10, 15), (3, 2, 1))])
def test_features_match_slow_computation(tuples):
    net = NTupleNet(tuples)
    grids, boards = random_boards(40, seed=2)
    feats = net.features(boards)
    syms = ntuple.symmetries(boards)
    assert feats.shape == (40, 8 * len(tuples))
    for k in range(40):
        for s in range(8):
            grid = g.exps(syms[s, k])
            for t, cells in enumerate(tuples):
                want = slow_index(grid, cells) + int(net.offsets[t])
                assert feats[k, s * len(tuples) + t] == want


def test_value_is_the_same_for_symmetric_boards():
    net = NTupleNet(ntuple.TUPLES_4)
    net.w[:] = np.random.default_rng(1).normal(size=net.w.shape).astype(np.float32)
    _, boards = random_boards(30, seed=3)
    syms = ntuple.symmetries(boards)
    base = net.value(boards)
    for s in range(8):
        assert np.allclose(net.value(syms[s]), base, rtol=1e-5)


def test_value_is_the_sum_of_table_entries():
    net = NTupleNet(ntuple.TUPLES_4)
    net.w[:] = np.arange(net.w.size, dtype=np.float32) % 7
    _, boards = random_boards(5, seed=4)
    feats = net.features(boards)
    assert np.allclose(net.value(boards), net.w[feats].sum(axis=1))


def test_update_averages_errors_of_repeated_weights():
    net = NTupleNet(ntuple.TUPLES_4)
    feats = np.array([[3, 3, 5], [5, 7, 7]], dtype=np.int64)
    net.update(feats, np.array([1.0, 2.0]), lr=0.5)
    assert net.w[3] == pytest.approx(0.5)       # ошибка 1 дважды -> среднее 1
    assert net.w[5] == pytest.approx(0.75)      # ошибки 1 и 2 -> среднее 1.5
    assert net.w[7] == pytest.approx(1.0)
    assert net.w[[0, 1, 2, 4, 6, 8]].sum() == 0


def test_big_batch_of_identical_positions_does_not_blow_up():
    # Тысяча одинаковых досок с одной ошибкой - шаг как у одной доски, а не в 1000 раз больше.
    net = NTupleNet(ntuple.TUPLES_4)
    _, boards = random_boards(1, seed=8)
    feats = np.repeat(net.features(boards), 1000, axis=0)
    net.update(feats, np.full(1000, 10.0), lr=0.01)
    assert 0 < net.value(boards)[0] <= 10.0 * 0.01 * net.n_features + 1e-3


def test_export_and_load_keep_values(tmp_path):
    net = NTupleNet(ntuple.TUPLES_4)
    rng = np.random.default_rng(5)
    mask = rng.random(net.w.size) < 0.1
    net.w[mask] = rng.normal(scale=3000, size=mask.sum()).astype(np.float32)
    path = tmp_path / "m.npz"
    net.export(path, meta={"games": 1})
    back = NTupleNet.load(path)
    assert back.tuples == net.tuples
    step = np.abs(net.w).max() / 32767
    assert np.abs(back.w - net.w).max() <= step
    _, boards = random_boards(20, seed=6)
    assert np.allclose(back.value(boards), net.value(boards), atol=step * net.n_features)


def test_raw_snapshot_round_trip(tmp_path):
    net = NTupleNet(ntuple.TUPLES_4)
    net.w[::97] = 1.25
    net.save_raw(tmp_path / "w.npy")
    back = NTupleNet.load_raw(tmp_path / "w.npy", ntuple.TUPLES_4)
    assert (back.w == net.w).all()
