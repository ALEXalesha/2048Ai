"""Где лежат ресурсы игры (модели). Данные игрока - в ctx.data_dir от aikit."""
import os
import sys

GAME_DIR = os.path.dirname(os.path.abspath(__file__))


def frozen():
    return getattr(sys, "frozen", False)


def resource_dir():
    """Папка пакета game: в собранной версии - внутри _MEIPASS."""
    if frozen():
        return os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)), "game")
    return GAME_DIR


def models_dir():
    return os.path.join(resource_dir(), "models")
