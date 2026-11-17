"""Iconclass codes: their English texts, their place in the tree, and the text embedded for each."""

import urllib.request
from functools import cache
from pathlib import Path

import iconclass

TEXTS = Path("cache/iconclass_texts_en.txt")
SOURCE = "https://raw.githubusercontent.com/iconclass/data/main/txt/en/{}.txt"
SOURCE_FILES = ("txt_en_0_1", "txt_en_2_3", "txt_en_4", "txt_en_5_6_7_8", "txt_en_9")


@cache
def texts() -> dict[str, str]:
    """Every code that has an English text, mapped to that text. Downloaded once."""
    if not TEXTS.exists():
        _download_texts()
    with TEXTS.open(encoding="utf-8") as f:
        pairs = (line.rstrip("\n").partition("|") for line in f)
        return {code: text for code, _, text in pairs if code and text}


def _download_texts():
    lines = set()
    for name in SOURCE_FILES:
        with urllib.request.urlopen(SOURCE.format(name), timeout=60) as response:
            lines.update(line.strip() for line in response.read().decode("utf-8").splitlines())
    TEXTS.parent.mkdir(parents=True, exist_ok=True)
    TEXTS.write_text("\n".join(sorted(line for line in lines if "|" in line)) + "\n", encoding="utf-8")


@cache
def tree():
    return iconclass.init()


def lineage(code: str) -> list[str]:
    """The code's ancestors and the code itself, top-level branch first: 14A1 gives 1, 14, 14A, 14A1.

    A few codes have a text but are missing from the iconclass package's tree; they get the
    lineage of their longest prefix that the tree has.
    """
    for end in range(len(code), 0, -1):
        try:
            steps = [str(node) for node in tree()[code[:end]].path()]
        except iconclass.NotationNotFound:
            continue
        return steps if steps[-1] == code else [*steps, code]
    return [code]


def document_text(code: str, variant: str) -> str:
    """What gets embedded for a code: its own text, its parent's text and its own, or every
    text along its lineage."""
    if variant == "standalone":
        return texts()[code]
    steps = lineage(code)[-2:] if variant == "parent" else lineage(code)
    return " → ".join(texts()[step] for step in steps if step in texts())
