import os
import sys

# Тесты делят ПК с другими агентами и обучением: потоков BLAS немного.
for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "2")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from aikit import offscreen  # noqa: E402

# Без экрана: Qt рисует в память. Ставится до любого импорта Qt.
offscreen.setup()

# Папка данных подменяется целиком (aikit.pick_data_root: <ПРОГРАММА>_HOME).
HOME_VAR = "2048AI_HOME"


@pytest.fixture(autouse=True)
def _own_data_dir(tmp_path, monkeypatch):
    """Каждый тест пишет настройки, статистику и место окна в свою временную папку,
    а не в профиль игрока."""
    home = tmp_path / "home"
    monkeypatch.setenv(HOME_VAR, str(home))
    return home


_app = None



@pytest.fixture
def qapp():
    """Одно QApplication на весь прогон: второе Qt создать не даст."""
    global _app
    from PySide6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication([])
    return _app
