"""2048 и ИИ, который сам научился в неё играть.

Вход игры по контракту ИИ-Игротеки: GAME_INFO и create_widget(ctx). Отдельный запуск -
main.py через aikit.run_standalone, оболочка Игротеки вставляет тот же виджет.
"""
from game.version import VERSION

# Имя программы: папка данных %APPDATA%\2048Ai, переменная 2048AI_HOME подменяет её.
APP_NAME = "2048Ai"

GAME_INFO = {
    "id": "game2048",
    "title": {"ru": "2048", "en": "2048"},
    "summary": {
        "ru": "Сдвигай плитки и собирай 2048 - или смотри, как ИИ, научившийся сам, доходит до 8192.",
        "en": "Slide tiles to reach 2048 - or watch a self-taught AI go for 8192.",
    },
    "preview": "docs/preview.png",
    "version": VERSION,
}


def create_widget(ctx):
    """Виджет игры. ctx - aikit.Context: тема, язык, настройки, статистика, папка данных."""
    from game.ui.widget import GameWidget
    return GameWidget(ctx)
