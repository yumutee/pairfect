"""Word lists bundled with pairfect, and their numeric encoding."""

import re
from importlib.resources import files
from pathlib import Path

import numpy as np

ANSWERS = "answers.txt"
GUESSES = "guesses.txt"

_WORD = re.compile(r"^[a-z]{5}$")


def load(path: str | Path | None, bundled: str) -> list[str]:
    """Read a whitespace-separated word list, or the bundled one if `path` is None."""
    if path is None:
        text = files("pairfect").joinpath("data", bundled).read_text()
    else:
        text = Path(path).read_text()
    words = sorted({w.lower() for w in text.split()})
    bad = [w for w in words if not _WORD.match(w)]
    if bad:
        raise ValueError(f"not five-letter words: {', '.join(bad[:5])}")
    return words


def encode(words: list[str]) -> np.ndarray:
    """Words as an (n, 5) uint8 array of letter indices"""
    raw = np.frombuffer("".join(words).encode("ascii"), np.uint8)
    return (raw.reshape(len(words), 5) - ord("a")).astype(np.uint8)
