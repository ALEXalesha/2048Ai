"""Поле 2048 с плавными сдвигами и слияниями.

Анимация хода идёт в две фазы: плитки едут из старых клеток в новые, потом слившиеся
плитки «подпрыгивают», а новая плитка вырастает. Время меряется часами, а не числом кадров:
на медленном ПК анимация не растягивается. duration 0 - без анимации (ИИ на «максимуме»).
"""
from __future__ import annotations

from PySide6.QtCore import QElapsedTimer, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from game.ui.tiles import board_colors, tile_colors

SLIDE_PART = 0.6           # доля времени на поездку плиток, остальное - на появление
SWIPE_MIN = 24             # пикселей мышью, чтобы считалось смахиванием
ARROW_DIRS = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}


class BoardView(QWidget):
    swiped = Signal(int)
    animation_finished = Signal()

    def __init__(self, size=4, theme="dark", parent=None):
        super().__init__(parent)
        self.size = size
        self.theme = theme
        self.grid = [[0] * size for _ in range(size)]
        self.moving = []          # (r0, c0, r1, c1, значение)
        self.popped = set()       # клетки слияний
        self.spawn = None         # (r, c, значение)
        self.duration = 0
        self.hint_dir = None
        self.caption = ""
        self.dimmed = False
        self._clock = QElapsedTimer()
        self._timer = QTimer(self)
        self._timer.setInterval(15)
        self._timer.timeout.connect(self._tick)
        self._press = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(240, 240)

    # --- состояние ---------------------------------------------------------------------

    def set_theme(self, theme):
        self.theme = theme
        self.update()

    def set_grid(self, grid):
        """Поставить поле сразу, без анимации."""
        self.finish_animation(emit=False)
        self.size = len(grid)
        self.grid = [row[:] for row in grid]
        self.update()

    def animate(self, new_grid, motions, spawn, duration_ms):
        """Показать ход: motions - (откуда, куда, значение, слилась ли) из generic.slide."""
        self.finish_animation(emit=False)
        self.grid = [row[:] for row in new_grid]
        self.moving = [(a[0], a[1], b[0], b[1], v) for a, b, v, _ in motions]
        self.popped = {b for _, b, _, merged in motions if merged}
        self.spawn = spawn
        self.duration = int(duration_ms)
        if self.duration <= 0:
            self.finish_animation()
            return
        self._clock.start()
        self._timer.start()
        self.update()

    def animating(self):
        return self._timer.isActive()

    def finish_animation(self, emit=True):
        was = self._timer.isActive()
        self._timer.stop()
        self.moving = []
        self.popped = set()
        self.spawn = None
        self.update()
        if emit and (was or self.duration <= 0):
            self.animation_finished.emit()

    def stop(self):
        self._timer.stop()

    def _progress(self):
        if not self._timer.isActive() or self.duration <= 0:
            return 1.0
        return min(1.0, self._clock.elapsed() / self.duration)

    def _tick(self):
        if self._progress() >= 1.0:
            self.finish_animation()
        else:
            self.update()

    def set_hint(self, direction):
        self.hint_dir = direction
        self.update()

    # --- геометрия ---------------------------------------------------------------------

    def board_rect(self):
        side = min(self.width(), self.height()) - 4
        # По горизонтали - по центру, по вертикали - к верху: подпись над полем рядом с ним.
        return QRectF((self.width() - side) / 2, 2, side, side)

    def cell_rect(self, r, c):
        br = self.board_rect()
        gap = br.width() * 0.12 / (self.size + 1)
        cell = (br.width() - gap * (self.size + 1)) / self.size
        return QRectF(br.x() + gap + c * (cell + gap), br.y() + gap + r * (cell + gap), cell, cell)

    def sizeHint(self):
        from PySide6.QtCore import QSize
        return QSize(460, 460)

    # --- рисование ---------------------------------------------------------------------

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        board, cell_color = board_colors(self.theme)
        br = self.board_rect()
        radius = br.width() * 0.03
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(board))
        p.drawRoundedRect(br, radius, radius)
        for r in range(self.size):
            for c in range(self.size):
                p.setBrush(QColor(cell_color))
                rect = self.cell_rect(r, c)
                p.drawRoundedRect(rect, rect.width() * 0.08, rect.width() * 0.08)

        t = self._progress()
        if self.moving and t < SLIDE_PART:
            k = _ease(t / SLIDE_PART)
            for r0, c0, r1, c1, v in self.moving:
                a, b = self.cell_rect(r0, c0), self.cell_rect(r1, c1)
                rect = QRectF(a.x() + (b.x() - a.x()) * k, a.y() + (b.y() - a.y()) * k, a.width(), a.height())
                self._tile(p, rect, v)
        else:
            phase = 1.0 if not self.moving else (t - SLIDE_PART) / (1 - SLIDE_PART)
            for r in range(self.size):
                for c in range(self.size):
                    v = self.grid[r][c]
                    if not v:
                        continue
                    rect = self.cell_rect(r, c)
                    scale = 1.0
                    if self.spawn and (r, c) == tuple(self.spawn[:2]) and phase < 1.0:
                        scale = 0.3 + 0.7 * _ease(phase)
                    elif (r, c) in self.popped and phase < 1.0:
                        scale = 1.0 + 0.14 * (1 - abs(2 * phase - 1))
                    if scale != 1.0:
                        cx, cy = rect.center().x(), rect.center().y()
                        w = rect.width() * scale
                        rect = QRectF(cx - w / 2, cy - w / 2, w, w)
                    self._tile(p, rect, v)

        if self.hint_dir is not None and not self.moving:
            self._arrow(p, self.hint_dir)
        if self.dimmed:
            veil = QColor(board)
            veil.setAlpha(150)
            p.setBrush(veil)
            p.drawRoundedRect(br, radius, radius)
        if self.caption:
            f = QFont(self.font())
            f.setPixelSize(max(14, int(br.width() * 0.07)))
            f.setBold(True)
            p.setFont(f)
            p.setPen(QColor(tile_colors(self.theme, 2)[1] if self.theme == "dark" else "#1b2030"))
            p.drawText(br, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.caption)
        p.end()

    def _tile(self, p, rect, value):
        bg, fg = tile_colors(self.theme, value)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(bg))
        p.drawRoundedRect(rect, rect.width() * 0.08, rect.width() * 0.08)
        text = str(value)
        f = QFont(self.font())
        f.setBold(True)
        share = {1: 0.5, 2: 0.46, 3: 0.4, 4: 0.32, 5: 0.26}.get(len(text), 0.24)
        f.setPixelSize(max(8, int(rect.height() * share)))
        p.setFont(f)
        p.setPen(QColor(fg))
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def _arrow(self, p, direction):
        """Стрелка подсказки поверх поля."""
        br = self.board_rect()
        dx, dy = ARROW_DIRS[direction]
        c = br.center()
        length = br.width() * 0.3
        tip = QPointF(c.x() + dx * length, c.y() + dy * length)
        tail = QPointF(c.x() - dx * length * 0.4, c.y() - dy * length * 0.4)
        color = QColor("#2dd4a8" if self.theme == "dark" else "#169c7a")
        color.setAlpha(220)
        pen = QPen(color, max(4.0, br.width() * 0.03))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawLine(tail, tip)
        head = br.width() * 0.07
        nx, ny = -dy, dx
        poly = QPolygonF([QPointF(tip.x() + dx * head * 0.6, tip.y() + dy * head * 0.6),
                          QPointF(tip.x() + nx * head - dx * head * 0.4, tip.y() + ny * head - dy * head * 0.4),
                          QPointF(tip.x() - nx * head - dx * head * 0.4, tip.y() - ny * head - dy * head * 0.4)])
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        path = QPainterPath()
        path.addPolygon(poly)
        p.drawPath(path)

    # --- мышь --------------------------------------------------------------------------

    def mousePressEvent(self, event):
        self.setFocus()
        self._press = event.position()

    def mouseReleaseEvent(self, event):
        if self._press is None:
            return
        d = event.position() - self._press
        self._press = None
        direction = swipe_direction(d.x(), d.y())
        if direction is not None:
            self.swiped.emit(direction)


def swipe_direction(dx, dy, minimum=SWIPE_MIN):
    """Направление смахивания по сдвигу мыши или None, если сдвиг мал."""
    if max(abs(dx), abs(dy)) < minimum:
        return None
    if abs(dx) > abs(dy):
        return 1 if dx > 0 else 3
    return 2 if dy > 0 else 0


def _ease(x):
    x = max(0.0, min(1.0, x))
    return 1 - (1 - x) ** 3
