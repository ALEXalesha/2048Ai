"""Сборка для Windows: папка dist/2048Ai, portable-архив и установщик NSIS.

    pip install pyinstaller pillow
    python build.py

Номер версии - один, в installer.nsi; по нему названы и установщик, и архив.
"""
import argparse
import os
import shutil
import subprocess
import sys
import zipfile

APP = "2048Ai"
BUILD = "build"
DIST = "dist"
ICON = os.path.join(BUILD, f"{APP}.ico")
NSIS_SCRIPT = "installer.nsi"
NSIS_PATHS = (r"C:\Program Files (x86)\NSIS\makensis.exe", r"C:\Program Files\NSIS\makensis.exe")
ROOT = os.path.abspath(os.path.dirname(__file__))

# Игре не нужны: без них сборка меньше. PIL нужен только build.py - для иконки.
EXCLUDE = ("torch", "torchvision", "scipy", "sklearn", "matplotlib", "PIL", "IPython", "notebook",
           "pandas", "pytest", "tkinter", "pygame")


def version():
    with open(os.path.join(ROOT, NSIS_SCRIPT), encoding="utf-8-sig") as f:
        for line in f:
            if line.startswith("!define VERSION"):
                return line.split('"')[1]
    raise SystemExit("в installer.nsi нет !define VERSION")


def make_icon(path=ICON):
    """Иконка рисуется тем же кодом, что в окне (game/ui/icon.py), и собирается в .ico."""
    sys.path.insert(0, ROOT)
    from aikit import offscreen
    offscreen.setup(force=True)
    from PIL import Image
    from PySide6.QtCore import QBuffer, QIODevice
    from PySide6.QtGui import QGuiApplication

    from game.ui.icon import render

    app = QGuiApplication.instance() or QGuiApplication([])  # noqa: F841 - нужен живым для QImage
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    render(256).save(buf, "PNG")
    import io
    image = Image.open(io.BytesIO(bytes(buf.data())))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    image.save(path, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return path


def run_pyinstaller(console=False):
    args = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
            "--name", APP, "--icon", os.path.join(ROOT, ICON),
            "--add-data", f"{os.path.join(ROOT, 'game', 'models')}{os.pathsep}{os.path.join('game', 'models')}",
            "--paths", ROOT,
            "--distpath", os.path.join(ROOT, DIST),
            "--workpath", os.path.join(ROOT, BUILD, "work"),
            "--specpath", os.path.join(ROOT, BUILD)]
    args += ["--console"] if console else ["--windowed"]
    for module in EXCLUDE:
        args += ["--exclude-module", module]
    args.append(os.path.join(ROOT, "main.py"))
    subprocess.run(args, check=True, cwd=ROOT)


def trim(folder):
    """Убрать из сборки то, что PyInstaller кладёт на всякий случай, а игре не нужно:
    программный OpenGL (окно рисуется без OpenGL), переводы Qt кроме русского, чтение PDF.
    Возвращает, сколько байт освободилось."""
    qt = os.path.join(folder, "_internal", "PySide6")
    doomed = [os.path.join(qt, "opengl32sw.dll"), os.path.join(qt, "Qt6Pdf.dll"),
              os.path.join(qt, "plugins", "imageformats", "qpdf.dll")]
    translations = os.path.join(qt, "translations")
    if os.path.isdir(translations):
        doomed += [os.path.join(translations, n) for n in os.listdir(translations) if not n.endswith("_ru.qm")]
    freed = 0
    for path in doomed:
        if os.path.isfile(path):
            freed += os.path.getsize(path)
            os.remove(path)
    return freed


def make_portable(folder=None):
    folder = folder or os.path.join(ROOT, DIST, APP)
    with open(os.path.join(folder, "portable.txt"), "w", encoding="utf-8") as f:
        f.write("Пока этот файл лежит рядом с 2048Ai.exe, настройки и статистика пишутся\n"
                "в папку saves рядом, а не в профиль пользователя. Удалите его, чтобы вернуть\n"
                "обычное поведение.\n")
    # Интерфейс по-русски, и подробный рассказ тоже в README.ru.md; README.md - английская выжимка.
    for doc in ("README.md", "README.ru.md", "LICENSE"):
        shutil.copy(os.path.join(ROOT, doc), os.path.join(folder, doc))

    archive = os.path.join(ROOT, DIST, f"{APP}-{version()}-portable.zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(folder):
            for name in files:
                full = os.path.join(root, name)
                zf.write(full, os.path.join(APP, os.path.relpath(full, folder)))
    return archive


def find_nsis():
    found = shutil.which("makensis")
    if found:
        return found
    return next((p for p in NSIS_PATHS if os.path.exists(p)), None)


def make_installer():
    nsis = find_nsis()
    if nsis is None:
        print("NSIS не найден, установщик пропущен")
        return None
    subprocess.run([nsis, NSIS_SCRIPT], check=True, cwd=ROOT)
    return os.path.join(ROOT, DIST, f"{APP}-{version()}-setup.exe")


def size_of(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, files in os.walk(path) for f in files)


def main():
    ap = argparse.ArgumentParser(description="Сборка portable-версии и установщика")
    ap.add_argument("--console", action="store_true", help="оставить окно консоли")
    ap.add_argument("--skip-installer", action="store_true")
    args = ap.parse_args()

    print("иконка...")
    make_icon(os.path.join(ROOT, ICON))
    print("PyInstaller...")
    run_pyinstaller(args.console)
    freed = trim(os.path.join(ROOT, DIST, APP))
    print(f"убрано лишнее: {freed / 1e6:.0f} МБ")
    print("portable-архив...")
    archive = make_portable()

    folder = os.path.join(ROOT, DIST, APP)
    print()
    print(f"папка      {folder}  ({size_of(folder) / 1e6:.0f} МБ)")
    print(f"portable   {archive}  ({size_of(archive) / 1e6:.0f} МБ)")

    if not args.skip_installer:
        print("установщик...")
        setup = make_installer()
        if setup and os.path.exists(setup):
            print(f"установщик {setup}  ({size_of(setup) / 1e6:.0f} МБ)")


if __name__ == "__main__":
    main()
