# 2048Ai

The 2048 puzzle plus an AI that taught itself to play it. Windows, Python, PySide6, numpy.

*Work in progress.* Русская версия: [README.ru.md](README.ru.md).

## How the AI works

An **n-tuple network** trained by **temporal-difference learning on afterstates**
(Szubert & Jaśkowski; Yeh, Matsuzaki and others): the board after the move, before the
new tile appears, is valued by summing lookup tables over a few groups of 6 cells, with
all 8 board symmetries. The network learns only by playing against chance. The strong
level adds **expectimax** search over every empty cell and both possible new tiles.

## Run from source

```
pip install -r requirements.txt
python main.py
```

Tests: `pip install -r requirements-dev.txt` and `python -m pytest -q`.

## License

MIT, see [LICENSE](LICENSE).
