"""N-tuple сеть: оценка поля 2048 суммой таблиц.

Сеть смотрит на несколько групп клеток («кортежей») - по умолчанию четыре группы по 6
клеток, как у Yeh и Jaśkowski. Степени плиток в группе дают номер строки таблицы, оценка
поля - сумма значений из таблиц по всем группам. Каждая группа прикладывается к полю во
всех 8 симметриях (повороты и отражения), таблица у симметричных копий общая: так сеть
сразу знает, что угол - это любой угол.

Это та же нейросеть с одним слоем, только вход у неё «один из 16^6» на группу, поэтому
вместо умножения матриц - выборка из таблицы. Всё на numpy, тысячи полей за раз.

Номер в таблице берётся прямо из битов поля: клетки группы, идущие подряд, - это подряд
идущие 4-битные поля числа, их достаточно сдвинуть и замаскировать.
"""
from __future__ import annotations

import numpy as np

from game.core.game2048 import transpose

U = np.uint64

# Клетки (номер = 4*строка + столбец). Две «полоски» 2x3 у края и две «уголка» из строк.
TUPLES_6 = (
    (0, 1, 2, 3, 4, 5),
    (4, 5, 6, 7, 8, 9),
    (0, 1, 2, 4, 5, 6),
    (4, 5, 6, 8, 9, 10),
)

# Маленькая сеть для быстрых тестов: строки и квадраты 2x2.
TUPLES_4 = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 4, 5),
    (1, 2, 5, 6),
    (5, 6, 9, 10),
)


def flip_h(b):
    """Отражение слева направо: столбец c -> 3 - c."""
    return (((b & U(0x000F000F000F000F)) << U(12)) | ((b & U(0x00F000F000F000F0)) << U(4))
            | ((b & U(0x0F000F000F000F00)) >> U(4)) | ((b & U(0xF000F000F000F000)) >> U(12)))


def flip_v(b):
    """Отражение сверху вниз: строка r -> 3 - r."""
    return (((b & U(0x000000000000FFFF)) << U(48)) | ((b & U(0x00000000FFFF0000)) << U(16))
            | ((b & U(0x0000FFFF00000000)) >> U(16)) | ((b & U(0xFFFF000000000000)) >> U(48)))


def symmetries(b):
    """Все 8 симметрий поля: массив (8, N)."""
    t = transpose(b)
    h, v, th, tv = flip_h(b), flip_v(b), flip_h(t), flip_v(t)
    return np.stack([b, h, v, flip_v(h), t, th, tv, flip_v(th)])


def _runs(cells):
    """Группа клеток -> куски подряд идущих клеток: (с какой клетки, сколько, куда в номере)."""
    runs = []
    j = 0
    while j < len(cells):
        k = j
        while k + 1 < len(cells) and cells[k + 1] == cells[k] + 1:
            k += 1
        runs.append((cells[j], k - j + 1, j))
        j = k + 1
    return runs


class NTupleNet:
    """Таблицы весов float32, одна на группу клеток, плюс правило выборки номеров."""

    def __init__(self, tuples=TUPLES_6, weights=None):
        self.tuples = tuple(tuple(int(c) for c in t) for t in tuples)
        self.sizes = [16 ** len(t) for t in self.tuples]
        self.offsets = np.cumsum([0] + self.sizes[:-1]).astype(np.int64)
        total = int(sum(self.sizes))
        if weights is None:
            weights = np.zeros(total, dtype=np.float32)
        if weights.shape != (total,):
            raise ValueError(f"весов {weights.shape}, нужно {total}")
        self.w = weights
        self._plan = []
        for t, off in zip(self.tuples, self.offsets):
            parts = [(U(4 * start), U((1 << (4 * n)) - 1), U(4 * at)) for start, n, at in _runs(t)]
            self._plan.append((int(off), parts))

    @property
    def n_features(self):
        """Сколько слагаемых в оценке одного поля: группы x 8 симметрий."""
        return 8 * len(self.tuples)

    def features(self, boards):
        """Номера весов для каждого поля: массив (N, групп * 8) int64."""
        boards = np.asarray(boards, dtype=np.uint64)
        syms = symmetries(boards)                          # (8, N)
        out = np.empty((boards.shape[0], 8, len(self.tuples)), dtype=np.int64)
        for k, (off, parts) in enumerate(self._plan):
            idx = np.zeros_like(syms)
            for shift, mask, at in parts:
                idx |= ((syms >> shift) & mask) << at
            out[:, :, k] = idx.T.astype(np.int64) + off
        return out.reshape(boards.shape[0], -1)

    def value(self, boards):
        """Оценка полей (после хода, до новой плитки): ожидаемые будущие очки."""
        boards = np.asarray(boards, dtype=np.uint64)
        if boards.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        return self.w[self.features(boards)].sum(axis=1, dtype=np.float64)

    def value_of_features(self, feats):
        return self.w[feats].sum(axis=1, dtype=np.float64)

    def update(self, feats, delta, lr):
        """TD-шаг: каждому весу, участвовавшему в оценке, добавить lr * (средняя ошибка).

        feats (M, F), delta (M,). Если вес встретился несколько раз - в разных партиях
        пачки или в разных симметриях одной доски, - ошибки усредняются, а не
        складываются. Иначе частые веса (пустая строка в начале партии) получали бы шаг,
        умноженный на размер пачки, и обучение разлеталось бы в бесконечность."""
        if feats.shape[0] == 0:
            return
        flat = feats.ravel()
        errors = np.repeat(np.asarray(delta, dtype=np.float64), feats.shape[1])
        uniq, inv = np.unique(flat, return_inverse=True)
        total = np.bincount(inv, weights=errors)
        count = np.bincount(inv)
        self.w[uniq] += (lr * total / count).astype(np.float32)

    # --- файлы --------------------------------------------------------------------------

    def save_raw(self, path):
        """Полные веса float32 - для снимков обучения и продолжения с места."""
        np.save(path, self.w)

    @classmethod
    def load_raw(cls, path, tuples=TUPLES_6):
        return cls(tuples, np.load(path))

    def export(self, path, meta=None):
        """Сжатая модель для игры: веса в int16 с масштабом на каждую группу.

        Точности хватает с запасом: шаг квантования - доли очка при оценках в тысячи, а
        непосещённые строки таблиц (нули) хорошо жмутся."""
        scales = []
        q = np.empty(self.w.shape, dtype=np.int16)
        for off, size in zip(self.offsets, self.sizes):
            part = self.w[off:off + size]
            peak = float(np.abs(part).max()) if size else 0.0
            scale = peak / 32767.0 if peak > 0 else 1.0
            q[off:off + size] = np.round(part / scale).astype(np.int16)
            scales.append(scale)
        lengths = np.array([len(t) for t in self.tuples], dtype=np.int64)
        flat = np.array([c for t in self.tuples for c in t], dtype=np.int64)
        np.savez_compressed(path, q=q, scales=np.array(scales, dtype=np.float64),
                            tuple_lengths=lengths, tuple_cells=flat,
                            meta=np.array(repr(meta or {})))

    @classmethod
    def load(cls, path):
        """Модель из export(): веса обратно во float32."""
        with np.load(path) as z:
            lengths = z["tuple_lengths"]
            flat = z["tuple_cells"]
            tuples, k = [], 0
            for n in lengths:
                tuples.append(tuple(int(c) for c in flat[k:k + n]))
                k += n
            net = cls(tuples)
            q = z["q"]
            for off, size, scale in zip(net.offsets, net.sizes, z["scales"]):
                net.w[off:off + size] = q[off:off + size].astype(np.float32) * np.float32(scale)
        return net
