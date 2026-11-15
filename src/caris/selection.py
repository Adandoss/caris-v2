"""Which codes an image gets: the top_k best, those scoring at least threshold, or both."""

import numpy as np

from caris.config import SelectionSettings


def ranking(scores: np.ndarray) -> np.ndarray:
    """Code positions from best to worst score. Equal scores keep index order, which is code order."""
    return np.argsort(-scores, kind="stable")


def cutoff(best_first: np.ndarray, top_k: int | None, threshold: float | None) -> int:
    """How many codes from the top of a best-first list of scores a rule keeps."""
    keep = len(best_first)
    if threshold is not None:
        keep = int(np.searchsorted(-best_first, -threshold, side="right"))
    return keep if top_k is None else min(keep, top_k)


def select(scores: np.ndarray, codes: list[str], rule: SelectionSettings) -> list[tuple[str, float]]:
    """The codes the rule keeps, with their scores, best first."""
    order = ranking(scores)
    keep = cutoff(scores[order], rule.top_k, rule.threshold)
    return [(codes[i], float(scores[i])) for i in order[:keep]]
