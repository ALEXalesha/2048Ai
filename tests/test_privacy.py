"""Личное не уходит в репозиторий: почта, фамилия, адрес домашнего Gitea, пути к папкам
на ПК автора. Сами строки здесь не записаны - берутся из настроек git, иначе тест
находил бы их в самом себе."""
import os
import re
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def secrets():
    found = []
    email = git("config", "user.email")
    if email:
        found.append(email)
        local = email.split("@")[0]
        # Фамилия - вторая часть имени ящика, без цифр.
        parts = re.split(r"[._]", local)
        if len(parts) > 1:
            surname = re.sub(r"\d+", "", parts[1])
            if len(surname) >= 4:
                found.append(surname)
    remote = git("remote", "get-url", "origin")
    m = re.match(r"^[a-z]+://([^/]+)/", remote)
    if m:
        found.append(m.group(1))
        found.append(m.group(1).split(":")[0])
    found.append("Drive" + "\\" + "Alexey")
    found.append("Drive" + "/" + "Alexey")
    return [s for s in found if s]


def tracked_files():
    out = git("ls-files")
    return [os.path.join(ROOT, p) for p in out.splitlines() if p]


def test_no_private_strings_in_tracked_files():
    files = tracked_files()
    if not files:
        pytest.skip("не git-репозиторий (например, копия без .git)")
    bad = []
    patterns = [s.lower() for s in secrets()]
    for path in files:
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            data = f.read()
        if b"\0" in data[:4096]:
            continue
        text = data.decode("utf-8", "replace").lower()
        for s in patterns:
            if s in text:
                bad.append((os.path.relpath(path, ROOT), "…" + s[-3:]))
    assert not bad, f"личное в файлах: {bad}"
