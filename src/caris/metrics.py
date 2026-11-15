"""Flat, hierarchical and ranking metrics, with the definitions of the previous system.

Flat metrics compare exact codes. Hierarchical metrics first add every ancestor to both sets
of codes (Kiritchenko et al., 2006), so a near miss earns partial credit. Ranking metrics
judge the order of a ranked list.
"""

from functools import cache

import iconclass

from caris.codes import tree

CUTOFFS = (1, 5, 10, 20)  # the k of precision@k and the other ranking metrics


def f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def flat_metrics(predicted: set[str], truth: set[str]) -> dict:
    hits = len(predicted & truth)
    precision = hits / len(predicted) if predicted else 0.0
    recall = hits / len(truth) if truth else 0.0
    return {"precision": precision, "recall": recall, "f1": f1(precision, recall)}


def ancestor_closure(codes) -> set[str]:
    """The codes and all their ancestors. A combined code such as 61B:31D14 counts as both parts."""
    return set().union(*(_closure(code) for code in codes if code))


@cache
def _closure(code: str) -> frozenset[str]:
    closure = set()
    for part in (part.strip() for part in iconclass.split_on_colon(code)):
        if not part:
            continue
        closure.add(part)
        try:
            closure.update(str(node) for node in tree()[part].path())
        except iconclass.NotationNotFound:
            # Not in the tree, e.g. a typo in the labels. Trim only the part before any
            # brackets, because partial bracket text is not a code.
            stem = part.split("(")[0].strip()
            while stem:
                closure.add(stem)
                stem = stem[:-1]
    return frozenset(closure)


def hierarchical_metrics(predicted, truth) -> dict:
    predicted, truth = ancestor_closure(predicted), ancestor_closure(truth)
    shared, either = predicted & truth, predicted | truth
    h_precision = len(shared) / len(predicted) if predicted else 0.0
    h_recall = len(shared) / len(truth) if truth else 0.0
    return {
        "h_precision": h_precision,
        "h_recall": h_recall,
        "h_f1": f1(h_precision, h_recall),
        "h_jaccard": len(shared) / len(either) if either else 0.0,
    }


def ranking_metrics(ranked: list[str], truth: list[str], cutoffs=CUTOFFS) -> dict:
    """Scores the order of a ranked list; truth must not be empty. precision@k divides by k
    even when fewer codes were ranked, so ranking fewer codes can't raise it."""
    truth_set = set(truth)
    result = {}
    for k in cutoffs:
        top = ranked[:k]
        hits = len(truth_set.intersection(top))
        result[f"precision@{k}"] = hits / k
        result[f"recall@{k}"] = hits / len(truth_set)
        result[f"hit@{k}"] = 1.0 if hits else 0.0
        result[f"h_f1@{k}"] = hierarchical_metrics(top, truth)["h_f1"]

    hits, reciprocal_rank, precision_sum = 0, 0.0, 0.0
    for rank, code in enumerate(ranked, 1):
        if code in truth_set:
            hits += 1
            precision_sum += hits / rank
            if hits == 1:
                reciprocal_rank = 1 / rank
            if hits == len(truth_set):
                break
    result["reciprocal_rank"] = reciprocal_rank
    result["average_precision"] = precision_sum / len(truth_set)
    return result
