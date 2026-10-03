"""Цвета плиток в духе тем Paint: от спокойного синего акцента через бирюзу и зелень к
тёплым и фиолетовым на больших плитках. Свои, не копия оригинальной 2048."""

TILES = {
    "dark": {
        "board": "#232833", "cell": "#2f3540",
        2: ("#3b4352", "#e6e8ec"), 4: ("#445069", "#e6e8ec"), 8: ("#3f6aa3", "#ffffff"),
        16: ("#4a7ab8", "#ffffff"), 32: ("#3f8fb0", "#ffffff"), 64: ("#2f9e9a", "#ffffff"),
        128: ("#2dae7f", "#ffffff"), 256: ("#6bb04a", "#ffffff"), 512: ("#b0a33a", "#ffffff"),
        1024: ("#d08a32", "#ffffff"), 2048: ("#e2683a", "#ffffff"), 4096: ("#d94a5a", "#ffffff"),
        8192: ("#c03a86", "#ffffff"), 16384: ("#8e4ac0", "#ffffff"), 32768: ("#5a4fd0", "#ffffff"),
    },
    "light": {
        "board": "#d3dae6", "cell": "#e9edf4",
        2: ("#ffffff", "#1b2030"), 4: ("#e3e9f5", "#1b2030"), 8: ("#a9c2ee", "#1b2030"),
        16: ("#7ea3e6", "#ffffff"), 32: ("#3a6fd8", "#ffffff"), 64: ("#2a8fc4", "#ffffff"),
        128: ("#1fa596", "#ffffff"), 256: ("#3fb06a", "#ffffff"), 512: ("#9db02a", "#ffffff"),
        1024: ("#e0a020", "#ffffff"), 2048: ("#e8742a", "#ffffff"), 4096: ("#e04848", "#ffffff"),
        8192: ("#c83a8a", "#ffffff"), 16384: ("#7a52d0", "#ffffff"), 32768: ("#4a3cc0", "#ffffff"),
    },
}


def tile_colors(theme, value):
    """(фон, текст) плитки. Больше 32768 не бывает, но на всякий случай - цвет 32768."""
    table = TILES.get(theme, TILES["dark"])
    v = min(int(value), 32768)
    return table.get(v, table[32768])


def board_colors(theme):
    table = TILES.get(theme, TILES["dark"])
    return table["board"], table["cell"]
