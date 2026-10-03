"""«Мысли ИИ»: оценка четырёх направлений стрелками разной длины и цвета.

Длина и цвет - от худшего разрешённого хода (короткая, красная) к лучшему (длинная,
зелёная). Запрещённый ход - серый кружок без стрелки.
"""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from aikit import theme as kit_theme

DIRS = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}


def arrow_shares(values):
    """Длина стрелок 0..1 и оценка -1..1 для цвета. Запрещённые ходы (-inf) - None."""
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return [None] * 4, [None] * 4
    lo, hi = min(finite), max(finite)
    lengths, heats = [], []
    for v in values:
        if v is None or not math.isfinite(v):
            lengths.append(None)
            heats.append(None)
            continue
        x = 1.0 if hi == lo else (v - lo) / (hi - lo)
        lengths.append(0.55 + 0.45 * x)
        heats.append(2 * x - 1)
    return lengths, heats


class Compass(QWidget):
    def __init__(self, theme="dark", parent=None):
        super().__init__(parent)
        self.theme = theme
        self.values = [None] * 4
        self.best = None
        self.setMinimumSize(150, 150)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(170)

    def sizeHint(self):
        return QSize(190, 170)

    def set_theme(self, theme):
        self.theme = theme
        self.update()

    def set_values(self, values, best=None):
        self.values = list(values) if values is not None else [None] * 4
        self.best = best
        self.update()

    def clear(self):
        self.set_values(None)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = kit_theme.palette(self.theme)
        side = min(self.width(), self.height())
        c = QPointF(self.width() / 2, self.height() / 2)
        reach = side * 0.42
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(pal["surface2"]))
        p.drawEllipse(c, side * 0.08, side * 0.08)
        lengths, heats = arrow_shares(self.values)
        for d, (dx, dy) in DIRS.items():
            if lengths[d] is None:
                p.setBrush(QColor(pal["border"]))
                p.drawEllipse(QPointF(c.x() + dx * reach * 0.5, c.y() + dy * reach * 0.5), 4, 4)
                continue
            color = QColor(*kit_theme.heat(heats[d], self.theme))
            if d == self.best:
                color = QColor(pal["good"])
            length = reach * lengths[d]
            start = QPointF(c.x() + dx * side * 0.1, c.y() + dy * side * 0.1)
            tip = QPointF(c.x() + dx * length, c.y() + dy * length)
            width = side * (0.07 if d == self.best else 0.05)
            pen = QPen(color, width)
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            p.setPen(pen)
            p.drawLine(start, QPointF(tip.x() - dx * width, tip.y() - dy * width))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color)
            nx, ny = -dy, dx
            h = width * 1.4
            p.drawPolygon(QPolygonF([QPointF(tip.x() + dx * h * 0.5, tip.y() + dy * h * 0.5),
                                     QPointF(tip.x() + nx * h - dx * h * 0.7, tip.y() + ny * h - dy * h * 0.7),
                                     QPointF(tip.x() - nx * h - dx * h * 0.7, tip.y() - ny * h - dy * h * 0.7)]))
        if all(v is None for v in lengths):
            f = QFont(self.font())
            f.setPixelSize(13)
            p.setFont(f)
            p.setPen(QColor(pal["muted"]))
            p.drawText(QRectF(0, 0, self.width(), self.height()), Qt.AlignmentFlag.AlignCenter, "?")
        p.end()
