"""Виджет игры: три режима, табло, «Мысли ИИ», статистика и настройки.

Режимы:
- «Играю сам» - стрелки, WASD или смахивание мышью; отмена хода; «Подсказка» показывает
  лучший ход стрелкой на поле и оценки всех направлений.
- «Играет ИИ» - пауза, скорость от «медленно» до «максимум», шаг по одному ходу.
- «Соревнование» - два поля с одним зерном случайности: у вас и у ИИ одни и те же новые
  плитки. ИИ ходит после каждого вашего хода; кто наберёт больше.

ИИ думает в отдельном потоке (AiRunner). Окно не создаёт своё главное окно: его даёт
aikit (отдельный запуск) или оболочка Игротеки.
"""
from __future__ import annotations

import math

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import (QCheckBox, QGridLayout, QHBoxLayout, QLabel, QScrollArea, QSlider,
                               QStackedWidget, QVBoxLayout, QWidget)

from aikit import settings as kit_settings
from aikit import theme as kit_theme
from aikit.i18n import Translator
from aikit.widgets import Combo, button, card, label, restyle, segmented
from game.ai import levels
from game.core.records import Records
from game.core.session import Session
from game.ui.ai_runner import AiRunner, next_tile_chance
from game.ui.board import BoardView
from game.ui.texts import TABLES
from game.ui.thoughts import Compass

MODES = ("play", "ai", "versus")
SPEEDS = ("slow", "normal", "fast", "max")
# (длительность анимации, пауза после хода) в мс для режима «Играет ИИ»
SPEED_TIMING = {"slow": (220, 520), "normal": (140, 140), "fast": (70, 0), "max": (0, 0)}
HUMAN_ANIM = 120
SIZES = (3, 4, 5, 6)
DIR_KEYS = {
    Qt.Key.Key_Up: 0, Qt.Key.Key_W: 0, Qt.Key.Key_Right: 1, Qt.Key.Key_D: 1,
    Qt.Key.Key_Down: 2, Qt.Key.Key_S: 2, Qt.Key.Key_Left: 3, Qt.Key.Key_A: 3,
}
DIR_NAMES = ("dir_up", "dir_right", "dir_down", "dir_left")
HINT_TIME = 0.4
SIDE_WIDTH = 372


def schema():
    return {
        "mode": ("play", kit_settings.choice(*MODES)),
        "level": ("strong", kit_settings.choice(*levels.LEVELS)),
        "speed": ("normal", kit_settings.choice(*SPEEDS)),
        "thoughts": (True, kit_settings.boolean()),
        "size": (4, kit_settings.choice(*SIZES)),
    }


def fit_segments(*groups):
    """Кнопки-переключатели шире жирного текста: выбранная кнопка пишется жирным, и
    ширина, посчитанная по обычному шрифту, обрезала бы её надпись."""
    for group in groups:
        for b in group.values():
            # Шрифт - как в теме (14px), жирный с запасом: у кнопки при постройке ещё
            # шрифт по умолчанию, тема применяется позже.
            f = QFont(b.font())
            f.setPixelSize(max(14, f.pixelSize()))
            f.setWeight(QFont.Weight.Bold)
            need = QFontMetrics(f).horizontalAdvance(b.text()) + 30
            b.setMinimumWidth(need)


def regrid(box, buttons, columns):
    """Переложить кнопки переключателя в сетку: в боковую панель ряд из четырёх не влезает.
    Старая раскладка уходит во временный виджет и удаляется вместе с ним."""
    old = box.layout()
    for b in buttons.values():
        old.removeWidget(b)                # кнопки остаются детьми box
    QWidget().setLayout(old)
    grid = QGridLayout(box)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setSpacing(6)
    for i, b in enumerate(buttons.values()):
        grid.addWidget(b, i // columns, i % columns)


def fmt_int(n):
    return f"{int(n):,}".replace(",", " ")


class GameWidget(QWidget):
    def __init__(self, ctx, brain=None, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.tr = Translator(TABLES, ctx.lang)
        self.settings = kit_settings.Settings(schema(), ctx.file("settings.json"))
        self.records = Records(ctx.file("records.json"))
        self.passport = levels.load_passport()
        self.runner = AiRunner(brain)
        self.runner.answered.connect(self._on_ai)
        self.gen = 0                    # номер партии/режима: ответы ИИ для старых отбрасываются
        self.paused = False
        self.stepping = False
        self.me = None
        self.ai = None
        self.versus_done = False
        self.last_values = None
        self.hint_shown = False
        self._closing = False
        self.ai_timer = QTimer(self)
        self.ai_timer.setSingleShot(True)
        self.ai_timer.timeout.connect(self._ai_request)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._build()
        self.ctx.changed.connect(self._on_ctx)
        self.retranslate()
        self.start_mode(self.settings["mode"])

    # === построение =====================================================================

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        top = QHBoxLayout()
        self.mode_box, self.mode_buttons = segmented({m: m for m in MODES}, self.start_mode)
        top.addWidget(self.mode_box)
        top.addStretch(1)
        self.nav_box, self.nav_buttons = segmented({"game": "", "stats": "", "settings": ""}, self.show_page)
        top.addWidget(self.nav_box)
        root.addLayout(top)

        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)
        self.stack.addWidget(self._build_game_page())
        self.stack.addWidget(self._build_stats_page())
        self.stack.addWidget(self._build_settings_page())
        self.nav_buttons["game"].setChecked(True)

    def _build_game_page(self):
        page = QWidget()
        page.setObjectName("page")
        row = QHBoxLayout(page)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(14)

        boards = QHBoxLayout()
        boards.setSpacing(14)
        self.me_col, self.me_title, self.me_board = self._board_column()
        self.ai_col, self.ai_title, self.ai_board = self._board_column()
        boards.addWidget(self.me_col, 1)
        boards.addWidget(self.ai_col, 1)
        self.me_board.swiped.connect(self.human_move)
        row.addLayout(boards, 1)

        side_inner = QWidget()
        side_inner.setObjectName("page")
        side = QVBoxLayout(side_inner)
        side.setContentsMargins(0, 0, 4, 0)
        side.setSpacing(10)

        score_card = card()
        grid = QGridLayout(score_card)
        grid.setContentsMargins(14, 12, 14, 12)
        self.score_caps, self.score_vals = {}, {}
        for i, key in enumerate(("score", "best", "max_tile")):
            cap = label("", "muted")
            val = label("0", "bigNumber")
            grid.addWidget(cap, 0, i)
            grid.addWidget(val, 1, i)
            self.score_caps[key], self.score_vals[key] = cap, val
        self.status = label("", "status", wrap=True)
        self.status.setMinimumHeight(28)
        grid.addWidget(self.status, 2, 0, 1, 3)
        side.addWidget(score_card)

        ctl = card()
        cl = QVBoxLayout(ctl)
        cl.setContentsMargins(14, 12, 14, 12)
        cl.setSpacing(8)
        r1 = QHBoxLayout()
        self.btn_new = button("", "primary", self.new_game)
        self.btn_undo = button("", None, self.undo)
        r1.addWidget(self.btn_new)
        r1.addWidget(self.btn_undo)
        cl.addLayout(r1)
        r2 = QHBoxLayout()
        self.btn_hint = button("", None, self.hint)
        self.btn_pause = button("", None, self.toggle_pause)
        self.btn_step = button("", None, self.step)
        r2.addWidget(self.btn_hint)
        r2.addWidget(self.btn_pause)
        r2.addWidget(self.btn_step)
        cl.addLayout(r2)
        self.speed_cap = label("", "muted")
        cl.addWidget(self.speed_cap)
        self.speed_box, self.speed_buttons = segmented({s: s for s in SPEEDS}, self.set_speed)
        regrid(self.speed_box, self.speed_buttons, columns=2)
        cl.addWidget(self.speed_box)
        self.level_cap = label("", "muted")
        cl.addWidget(self.level_cap)
        self.level_combo = Combo()
        for lv in levels.LEVELS:
            self.level_combo.addItem(lv, lv)
        self.level_combo.currentIndexChanged.connect(self._on_level)
        cl.addWidget(self.level_combo)
        side.addWidget(ctl)

        th = card()
        tl = QVBoxLayout(th)
        tl.setContentsMargins(14, 12, 14, 12)
        tl.setSpacing(6)
        self.thoughts_title = label("", "section")
        tl.addWidget(self.thoughts_title)
        self.thoughts_box = QCheckBox()
        self.thoughts_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.thoughts_box.setChecked(self.settings["thoughts"])
        self.thoughts_box.toggled.connect(self._on_thoughts)
        tl.addWidget(self.thoughts_box)
        self.compass = Compass(self.ctx.theme)
        tl.addWidget(self.compass)
        self.values_label = label("", "hint", wrap=True)
        tl.addWidget(self.values_label)
        self.expected_label = label("", None, wrap=True)
        tl.addWidget(self.expected_label)
        self.chance_label = label("", None, wrap=True)
        tl.addWidget(self.chance_label)
        self.passport_note = label("", "hint", wrap=True)
        tl.addWidget(self.passport_note)
        side.addWidget(th)

        self.note = label("", "hint", wrap=True)
        side.addWidget(self.note)
        side.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(side_inner)
        scroll.setFixedWidth(SIDE_WIDTH)
        side_inner.setMaximumWidth(SIDE_WIDTH - 14)
        row.addWidget(scroll)
        return page

    def _board_column(self):
        col = QWidget()
        lay = QVBoxLayout(col)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        title = label("", "section")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(title)
        board = BoardView(4, self.ctx.theme)
        lay.addWidget(board, 1)
        return col, title, board

    def _build_stats_page(self):
        page = QWidget()
        page.setObjectName("page")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(4, 4, 4, 4)
        self.stats_title = label("", "heading")
        lay.addWidget(self.stats_title)
        self.stats_size = label("", "section")
        lay.addWidget(self.stats_size)
        box = card()
        self.stats_grid = QGridLayout(box)
        self.stats_grid.setContentsMargins(16, 12, 16, 12)
        self.stats_grid.setHorizontalSpacing(30)
        lay.addWidget(box)
        self.stats_versus = label("", None, wrap=True)
        lay.addWidget(self.stats_versus)
        self.stats_passport = label("", None, wrap=True)
        lay.addWidget(self.stats_passport)
        lay.addStretch(1)
        return page

    def _build_settings_page(self):
        page = QWidget()
        page.setObjectName("page")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(10)
        self.settings_title = label("", "heading")
        lay.addWidget(self.settings_title)
        self.theme_cap = label("", "section")
        lay.addWidget(self.theme_cap)
        self.theme_box, self.theme_buttons = segmented({t: t for t in kit_theme.THEMES}, self.ctx.set_theme)
        lay.addWidget(self.theme_box, 0, Qt.AlignmentFlag.AlignLeft)
        self.lang_cap = label("", "section")
        lay.addWidget(self.lang_cap)
        self.lang_box, self.lang_buttons = segmented({"ru": "Русский", "en": "English"}, self.ctx.set_lang)
        lay.addWidget(self.lang_box, 0, Qt.AlignmentFlag.AlignLeft)
        self.size_cap = label("", "section")
        lay.addWidget(self.size_cap)
        self.size_box, self.size_buttons = segmented({s: f"{s}×{s}" for s in SIZES}, self.set_size)
        lay.addWidget(self.size_box, 0, Qt.AlignmentFlag.AlignLeft)
        self.size_note = label("", "hint", wrap=True)
        lay.addWidget(self.size_note)
        self.volume_cap = label("", "section")
        lay.addWidget(self.volume_cap)
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.volume.setMaximumWidth(320)
        self.volume.setValue(int(getattr(self.ctx.sound, "volume", 70)))
        self.volume.valueChanged.connect(self.ctx.set_volume)
        lay.addWidget(self.volume)
        lay.addStretch(1)
        return page

    # === язык, тема ===================================================================

    def _on_ctx(self, what):
        if what == "lang":
            self.tr.lang = self.ctx.lang
            self.retranslate()
        elif what == "theme":
            self.apply_theme()

    def apply_theme(self):
        for b in (self.me_board, self.ai_board):
            b.set_theme(self.ctx.theme)
        self.compass.set_theme(self.ctx.theme)
        self.level_combo.set_arrow_color(kit_theme.palette(self.ctx.theme)["text"])
        for t, b in self.theme_buttons.items():
            b.setChecked(t == self.ctx.theme)

    def retranslate(self):
        t = self.tr
        for m, b in self.mode_buttons.items():
            b.setText(t(f"mode_{m}"))
        for k, b in self.nav_buttons.items():
            b.setText(t(f"nav_{k}"))
        for k, cap in self.score_caps.items():
            cap.setText(t(k))
        self.btn_new.setText(t("new_game"))
        self.btn_undo.setText(t("undo"))
        self.btn_hint.setText(t("hint"))
        self.btn_step.setText(t("step"))
        self.speed_cap.setText(t("speed"))
        for s, b in self.speed_buttons.items():
            b.setText(t(f"speed_{s}"))
        self.level_cap.setText(t("level"))
        for i, lv in enumerate(levels.LEVELS):
            self.level_combo.setItemText(i, t(f"level_{lv}"))
        self.thoughts_title.setText(t("thoughts"))
        self.thoughts_box.setText(t("thoughts_toggle"))
        self.me_title.setText(t("you"))
        self.ai_title.setText(t("ai"))
        self.settings_title.setText(t("settings_title"))
        self.theme_cap.setText(t("theme"))
        for th, b in self.theme_buttons.items():
            b.setText(t(f"theme_{th}"))
        for lg, b in self.lang_buttons.items():
            b.setChecked(lg == self.ctx.lang)
        self.lang_cap.setText(t("language"))
        self.size_cap.setText(t("board_size"))
        self.size_note.setText(t("size_note"))
        self.volume_cap.setText(t("volume"))
        self.apply_theme()
        fit_segments(self.mode_buttons, self.nav_buttons, self.speed_buttons, self.theme_buttons,
                     self.lang_buttons, self.size_buttons)
        self._sync_controls()
        self.refresh()
        self.refresh_stats()

    # === режимы и партии ================================================================

    @property
    def mode(self):
        return self.settings["mode"]

    @property
    def size(self):
        return self.settings["size"]

    @property
    def level(self):
        return self.settings["level"]

    def start_mode(self, mode):
        if mode not in MODES:
            return
        self.settings.set("mode", mode)
        self.new_game()

    def new_game(self):
        self.gen += 1
        self.runner.cancel()
        self.ai_timer.stop()
        self.paused = False
        self.versus_done = False
        self.hint_shown = False
        self.last_values = None
        self.compass.clear()
        size = self.size
        if self.mode == "play":
            self.me, self.ai = Session(size, undo=True), None
        elif self.mode == "ai":
            self.me, self.ai = None, Session(size)
        else:
            seed = Session(size).seed
            self.me, self.ai = Session(size, seed=seed), Session(size, seed=seed)
        self.me_col.setVisible(self.me is not None)
        self.ai_col.setVisible(self.ai is not None)
        # Подписи «Вы» и «ИИ» нужны только там, где полей два.
        self.me_title.setVisible(self.mode == "versus")
        self.ai_title.setVisible(self.mode == "versus")
        if self.me:
            self.me_board.set_grid(self.me.grid)
            self.me_board.set_hint(None)
        if self.ai:
            self.ai_board.set_grid(self.ai.grid)
            self.ai_board.set_hint(None)
        for b in (self.me_board, self.ai_board):
            b.caption = ""
            b.dimmed = False
        self._sync_controls()
        self.refresh()
        if self.mode == "ai":
            self._schedule_ai(0)
        elif self.mode == "play":
            self._ask_thoughts()
        if self.me is not None:
            self.me_board.setFocus()

    def _sync_controls(self):
        mode = self.mode
        for m, b in self.mode_buttons.items():
            b.setChecked(m == mode)
        self.btn_undo.setVisible(mode == "play")
        self.btn_hint.setVisible(mode == "play")
        self.btn_pause.setVisible(mode == "ai")
        self.btn_step.setVisible(mode == "ai")
        self.speed_cap.setVisible(mode == "ai")
        self.speed_box.setVisible(mode == "ai")
        for s, b in self.speed_buttons.items():
            b.setChecked(s == self.settings["speed"])
        self.btn_pause.setText(self.tr("resume") if self.paused else self.tr("pause"))
        self.btn_step.setEnabled(self.paused)
        self.btn_undo.setEnabled(bool(self.me and self.me.can_undo()))
        self.level_combo.blockSignals(True)
        self.level_combo.setCurrentIndex(levels.LEVELS.index(self.level))
        self.level_combo.blockSignals(False)
        ai_real = self.size == 4 and self.runner.brain.available(self.level)
        self.level_combo.setEnabled(self.size == 4)
        for s, b in self.size_buttons.items():
            b.setChecked(s == self.size)
        notes = []
        if self.size != 4:
            notes.append(self.tr("other_size"))
        elif not ai_real:
            notes.append(self.tr("no_model"))
        notes.append(self.tr({"play": "keys_hint", "ai": "keys_hint", "versus": "versus_hint"}[mode]))
        if mode == "ai":
            notes.pop()
        self.note.setText("\n\n".join(notes))

    def set_speed(self, speed):
        self.settings.set("speed", speed)
        self._sync_controls()

    def set_size(self, size):
        if size not in SIZES:
            return
        self.settings.set("size", size)
        self.new_game()
        self.refresh_stats()

    def _on_level(self, index):
        if 0 <= index < len(levels.LEVELS):
            self.settings.set("level", levels.LEVELS[index])
            self.new_game()

    def _on_thoughts(self, on):
        self.settings.set("thoughts", bool(on))
        if on:
            self._ask_thoughts()
        self.refresh()

    def show_page(self, name):
        self.stack.setCurrentIndex({"game": 0, "stats": 1, "settings": 2}[name])
        if name == "stats":
            self.refresh_stats()
        for k, b in self.nav_buttons.items():
            b.setChecked(k == name)

    # === ход человека ===================================================================

    def keyPressEvent(self, event):
        d = DIR_KEYS.get(event.key())
        if d is None:
            if event.key() == Qt.Key.Key_Z and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.undo()
                return
            super().keyPressEvent(event)
            return
        self.human_move(d)

    def human_move(self, direction):
        if self.me is None or self.me.over() or self._closing:
            return False
        if self.mode == "versus" and (self.runner.busy() or self.ai_board.animating()):
            return False                      # ИИ ещё не сделал свой ответный ход
        res = self.me.move(direction)
        if not res.moved:
            return False
        self.hint_shown = False
        self.me_board.set_hint(None)
        self.me_board.animate(self.me.grid, res.motions, res.spawn, HUMAN_ANIM)
        self.ctx.sound.play("capture" if any(m[3] for m in res.motions) else "move")
        if res.reached:
            self.ctx.sound.play("win")
        if res.over:
            self._game_over_me()
        if self.mode == "versus":
            if self.ai is not None and not self.ai.over():
                self._ask_ai_move()
            self._check_versus()
        elif self.mode == "play":
            self._ask_thoughts()
        self._sync_controls()
        self.refresh(reached=res.reached)
        return True

    def undo(self):
        if self.mode != "play" or self.me is None or not self.me.undo():
            return False
        self.me_board.set_grid(self.me.grid)
        self.me_board.caption = ""
        self.me_board.dimmed = False
        self.hint_shown = False
        self.me_board.set_hint(None)
        self._sync_controls()
        self.refresh()
        self._ask_thoughts()
        return True

    def hint(self):
        if self.mode != "play" or self.me is None or self.me.over():
            return
        self.hint_shown = True
        self._ask(self.me, "hint", depth=3, time_limit=HINT_TIME)

    def _game_over_me(self):
        if not self.me.finished_recorded:
            self.me.finished_recorded = True
            self.records.add("me", self.size, self.me.score, self.me.max_tile())
        if self.mode == "play":
            self.ctx.sound.play("loss")
            self.me_board.dimmed = True
            self.me_board.caption = self.tr("status_over", score=fmt_int(self.me.score))

    # === ИИ ============================================================================

    def _ask(self, session, purpose, **extra):
        job = {"gen": self.gen, "purpose": purpose, "grid": [row[:] for row in session.grid],
               "level": self.level, **extra}
        return self.runner.ask(job)

    def _ask_thoughts(self):
        if self.mode == "play" and self.settings["thoughts"] and self.me and not self.me.over():
            self._ask(self.me, "thoughts")

    def _ask_ai_move(self):
        self._ask(self.ai, "move")

    def _schedule_ai(self, delay):
        if self.mode == "ai" and not self.paused and self.ai and not self.ai.over():
            self.ai_timer.start(max(0, int(delay)))

    def _ai_request(self):
        if self._closing or self.ai is None or self.ai.over() or self.runner.busy() or self.paused:
            return
        self._ask_ai_move()
        self.refresh()

    def toggle_pause(self):
        if self.mode != "ai":
            return
        self.paused = not self.paused
        if self.paused:
            self.ai_timer.stop()
        else:
            self._schedule_ai(0)
        self._sync_controls()
        self.refresh()

    def step(self):
        """Один ход ИИ на паузе."""
        if self.mode != "ai" or not self.paused or self.ai is None or self.ai.over():
            return
        if self.runner.busy():
            return
        self.stepping = True
        self._ask_ai_move()

    def _on_ai(self, job, result):
        if self._closing or job.get("gen") != self.gen:
            return
        purpose = job["purpose"]
        self.last_values = (result.get("values"), result.get("move"), result.get("depth", 0),
                            result.get("fallback", False), job["grid"])
        if purpose in ("hint", "thoughts"):
            if self.me is None or job["grid"] != self.me.grid:
                return
            self.compass.set_values(result["values"], result["move"])
            if purpose == "hint" and result["move"] is not None:
                self.me_board.set_hint(result["move"])
            self.refresh()
            return
        # ход ИИ
        session = self.ai
        if session is None or job["grid"] != session.grid:
            return
        move = result.get("move")
        if self.settings["thoughts"] or self.mode == "versus":
            self.compass.set_values(result["values"], move)
        if move is None:
            self._ai_over()
            return
        res = session.move(move)
        anim, pause = SPEED_TIMING[self.settings["speed"]]
        if self.mode == "versus":
            anim, pause = HUMAN_ANIM, 0
            if self.me is not None and self.me.over():
                anim, pause = SPEED_TIMING["fast"]
        self.ai_board.animate(session.grid, res.motions, res.spawn, anim)
        if res.over:
            self._ai_over()
        elif self.mode == "ai" and not self.paused:
            self._schedule_ai(anim + pause)
        elif self.mode == "versus" and self.me is not None and self.me.over():
            QTimer.singleShot(anim, self._versus_finish_step)
        self.stepping = False
        if self.mode == "versus":
            self._check_versus()
        self.refresh()

    def _versus_finish_step(self):
        if self.mode == "versus" and self.ai and not self.ai.over() and not self.runner.busy():
            self._ask_ai_move()

    def _ai_over(self):
        s = self.ai
        if not s.finished_recorded:
            s.finished_recorded = True
            if self.size == 4 and self.runner.brain.available(self.level):
                self.records.add("ai", self.size, s.score, s.max_tile())
        if self.mode == "ai":
            self.ai_board.dimmed = True
            self.ai_board.caption = self.tr("status_ai_over", score=fmt_int(s.score), tile=s.max_tile())
        self._check_versus()
        self.refresh()

    def _check_versus(self):
        if self.mode != "versus" or self.versus_done or self.me is None or self.ai is None:
            return
        if self.me.over() and not self.ai.over() and not self.runner.busy():
            self._versus_finish_step()
        if not (self.me.over() and self.ai.over()):
            return
        self.versus_done = True
        me, ai = self.me.score, self.ai.score
        result = "win" if me > ai else ("loss" if me < ai else "draw")
        self.ctx.stats.record(f"versus/{self.size}/{self.level}", result)
        self.ctx.sound.play(result)
        for b in (self.me_board, self.ai_board):
            b.dimmed = True
        self.me_board.caption = self.tr(f"versus_{result}", me=fmt_int(me), ai=fmt_int(ai))
        self.refresh()

    # === табло и мысли ==================================================================

    def current(self):
        return self.me if self.mode in ("play", "versus") else self.ai

    def refresh(self, reached=None):
        s = self.current()
        t = self.tr
        if s is not None:
            self.score_vals["score"].setText(fmt_int(s.score))
            self.score_vals["best"].setText(fmt_int(max(self.records.best_score(self.size), s.score)))
            self.score_vals["max_tile"].setText(str(s.max_tile()))
        status, name = "", "status"
        if self.mode == "play" and self.me:
            if self.me.over():
                status, name = t("status_over", score=fmt_int(self.me.score)), "statusLoss"
            elif self.me.reached:
                status, name = t("status_reached", tile=self.me.reached), "statusWin"
            else:
                status = t("status_play")
        elif self.mode == "ai" and self.ai:
            if self.ai.over():
                status = t("status_ai_over", score=fmt_int(self.ai.score), tile=self.ai.max_tile())
            elif self.paused:
                status = t("status_ai_paused")
            else:
                status = t("status_ai_thinking") if self.runner.busy() else ""
        elif self.mode == "versus" and self.me and self.ai:
            if self.versus_done:
                me, ai = self.me.score, self.ai.score
                result = "win" if me > ai else ("loss" if me < ai else "draw")
                status = t(f"versus_{result}", me=fmt_int(me), ai=fmt_int(ai))
                name = {"win": "statusWin", "loss": "statusLoss", "draw": "statusDraw"}[result]
            elif self.me.over():
                status = t("status_versus_ai_finishing")
            else:
                status = t("status_versus")
            self.me_title.setText(f"{t('you')}: {fmt_int(self.me.score)}")
            self.ai_title.setText(f"{t('ai')}: {fmt_int(self.ai.score)}")
        if self.mode != "versus":
            self.me_title.setText(t("you"))
            self.ai_title.setText(t("ai"))
        self.status.setText(status)
        if self.status.objectName() != name:
            restyle(self.status, name)
        self._refresh_thoughts()

    def _refresh_thoughts(self):
        t = self.tr
        show = self.settings["thoughts"] or self.hint_shown or self.mode == "versus"
        self.compass.setVisible(show)
        if not show or self.last_values is None:
            self.values_label.setText("" if show else t("thoughts_off"))
            self.expected_label.setText("")
            self.chance_label.setText("")
            self.passport_note.setText("")
            return
        values, move, depth, fallback, grid = self.last_values
        lines = []
        if values and not fallback:
            for d, v in enumerate(values):
                txt = "-" if v is None or not math.isfinite(v) else fmt_int(round(v))
                mark = " ←" if d == move else ""
                lines.append(f"{t(DIR_NAMES[d])}: {txt}{mark}")
            lines.append(t("depth", depth=depth))
        self.values_label.setText("\n".join(lines))
        s = self.current() if self.mode != "versus" else self.ai
        if s is None or fallback or not values:
            self.expected_label.setText("")
            self.chance_label.setText("")
            self.passport_note.setText("")
            return
        finite = [v for v in values if v is not None and math.isfinite(v)]
        if finite:
            self.expected_label.setText(t("expected", score=fmt_int(s.score + max(finite))))
        top = max(max(row) for row in grid)
        tile, chance = next_tile_chance(self.passport, top)
        if chance is None:
            self.chance_label.setText(t("next_tile_unknown", tile=tile))
        else:
            self.chance_label.setText(t("next_tile", tile=tile, chance=f"{100 * chance:.0f} %"))
        self.passport_note.setText(t("by_passport"))

    def refresh_stats(self):
        t = self.tr
        self.stats_title.setText(t("stats_title"))
        self.stats_size.setText(t("stats_size", size=self.size))
        while self.stats_grid.count():
            item = self.stats_grid.takeAt(0)
            w = item.widget()
            if w:
                # Спрятать сразу: deleteLater удалит лишь в следующем цикле событий, и до
                # этого старые подписи рисовались бы поверх новых.
                w.hide()
                w.setParent(None)
                w.deleteLater()
        me, ai = self.records.get("me", self.size), self.records.get("ai", self.size)
        self.stats_grid.addWidget(label("", "muted"), 0, 0)
        self.stats_grid.addWidget(label(t("you"), "section"), 0, 1)
        self.stats_grid.addWidget(label(t("ai"), "section"), 0, 2)
        rows = [("stats_games", lambda e: fmt_int(e["games"])),
                ("stats_best", lambda e: fmt_int(e["best_score"])),
                ("stats_avg", lambda e: fmt_int(e["total_score"] / e["games"]) if e["games"] else "-"),
                ("stats_tile", lambda e: str(e["best_tile"]) if e["best_tile"] else "-"),
                ("stats_2048", lambda e: fmt_int(e["reached_2048"]))]
        for i, (key, f) in enumerate(rows, start=1):
            self.stats_grid.addWidget(label(t(key), "muted"), i, 0)
            self.stats_grid.addWidget(label(f(me)), i, 1)
            self.stats_grid.addWidget(label(f(ai)), i, 2)
        lines = []
        for lv in levels.LEVELS:
            e = self.ctx.stats.entry(f"versus/{self.size}/{lv}")
            if e["games"]:
                lines.append(t("stats_versus", level=t(f"level_{lv}")) + ": "
                             + t("stats_versus_line", wins=e["wins"], losses=e["losses"], draws=e["draws"]))
        self.stats_versus.setText("\n".join(lines))
        res = (self.passport or {}).get("results") or {}
        if not res:
            self.stats_passport.setText(t("stats_passport_none"))
            return
        out = [t("stats_passport", games=self.passport.get("eval", {}).get("games", "?"))]
        for name in ("random", "corner") + levels.LEVELS:
            r = res.get(name)
            if not r:
                continue
            title = t(f"level_{name}" if name in levels.LEVELS else f"bot_{name}")
            pc = {k: f"{100 * r['reached'].get(k, 0):.1f} %" for k in ("2048", "4096", "8192")}
            out.append(t("stats_passport_row", level=title, p2048=pc["2048"], p4096=pc["4096"],
                         p8192=pc["8192"], avg=fmt_int(r["avg_score"])))
        self.stats_passport.setText("\n".join(out))

    # === закрытие ======================================================================

    def shutdown(self):
        """Остановить таймеры, анимации и поток ИИ - не дольше 2 секунд."""
        self._closing = True
        self.ai_timer.stop()
        self.me_board.stop()
        self.ai_board.stop()
        return self.runner.stop(2000)

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)
