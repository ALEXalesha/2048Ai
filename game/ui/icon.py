"""Иконка программы, нарисованная кодом: поле 2x2 с плитками 2, 0, 4 и большой 2048-й
в углу. Тем же кодом её собирает build.py в .ico."""
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPixmap

from game.ui.tiles import tile_colors


def render(size=256):
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("#232833"))
    p.drawRoundedRect(QRectF(0, 0, size, size), size * 0.18, size * 0.18)
    gap = size * 0.07
    cell = (size - 3 * gap) / 2
    tiles = [(0, 0, 2), (1, 0, 8), (0, 1, 64), (1, 1, 2048)]
    for c, r, v in tiles:
        rect = QRectF(gap + c * (cell + gap), gap + r * (cell + gap), cell, cell)
        bg, fg = tile_colors("dark", v)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(bg))
        p.drawRoundedRect(rect, cell * 0.14, cell * 0.14)
        if size >= 48:
            f = QFont("Segoe UI")
            f.setBold(True)
            f.setPixelSize(int(cell * (0.5 if v < 100 else 0.27)))
            p.setFont(f)
            p.setPen(QColor(fg))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(v))
    p.end()
    return img


def icon():
    ic = QIcon()
    for s in (16, 24, 32, 48, 64, 128, 256):
        ic.addPixmap(QPixmap.fromImage(render(s)))
    return ic
