"""Сборка: номер версии в одном месте с установщиком, модели в сборке, иконка, скрипт
публикации без личного."""
import os
import re

import pytest

import build
from game.version import VERSION

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def nsi():
    with open(os.path.join(ROOT, "installer.nsi"), encoding="utf-8-sig") as f:
        return f.read()


def test_version_is_the_same_in_the_game_and_the_installer():
    assert build.version() == VERSION


def test_installer_registers_under_the_app_name_without_admin_rights():
    text = nsi()
    assert re.search(r'!define APP "2048Ai"', text)
    assert 'OutFile "dist\\${APP}-${VERSION}-setup.exe"' in text
    assert "RequestExecutionLevel user" in text
    assert 'Delete "$INSTDIR\\portable.txt"' in text
    # Данные игрока - в %APPDATA%\2048Ai (aikit), их и предлагает удалить деинсталлятор.
    assert 'RMDir /r "$APPDATA\\${APP}"' in text


def test_models_are_packed_into_the_build():
    src = open(os.path.join(ROOT, "build.py"), encoding="utf-8").read()
    assert "'game', 'models'" in src


def test_icon_is_drawn_in_every_size(tmp_path, qapp):
    Image = pytest.importorskip("PIL.Image")
    path = build.make_icon(str(tmp_path / "icon.ico"))
    with Image.open(path) as ico:
        assert {(16, 16), (32, 32), (256, 256)} <= set(ico.info["sizes"])


def test_trim_removes_only_what_the_game_does_not_need(tmp_path):
    qt = tmp_path / "_internal" / "PySide6"
    (qt / "translations").mkdir(parents=True)
    (qt / "plugins" / "imageformats").mkdir(parents=True)
    keep = [qt / "Qt6Core.dll", qt / "translations" / "qtbase_ru.qm", qt / "plugins" / "imageformats" / "qico.dll"]
    gone = [qt / "opengl32sw.dll", qt / "Qt6Pdf.dll", qt / "translations" / "qtbase_de.qm",
            qt / "plugins" / "imageformats" / "qpdf.dll"]
    for p in keep + gone:
        p.write_bytes(b"12345")
    assert build.trim(str(tmp_path)) == 5 * len(gone)
    assert all(p.exists() for p in keep)
    assert not any(p.exists() for p in gone)


def test_publish_script_takes_private_strings_from_git_not_from_the_file():
    # Выкладка - общий PythonAiGames/tools/publish_game.sh: почту и адрес Gitea он берёт из
    # git config и origin. Здесь только обёртка и настройки - и личного в них нет.
    wrap = open(os.path.join(ROOT, "tools", "publish_github.sh"), encoding="utf-8").read()
    conf = open(os.path.join(ROOT, "tools", "publish.conf"), encoding="utf-8").read()
    assert "PythonAiGames/tools/publish_game.sh" in wrap
    assert "REPO=ALEXalesha/2048Ai" in conf
    for text in (wrap, conf):
        assert "@gmail" not in text and "192.168." not in text
