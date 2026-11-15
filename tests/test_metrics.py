"""Hand-made checks of what ends up in the thesis: the metrics, and how codes are selected."""

import numpy as np
import pytest

from caris.config import SelectionSettings
from caris.metrics import ancestor_closure, flat_metrics, hierarchical_metrics, ranking_metrics
from caris.selection import cutoff, select


def test_flat_metrics_count_exact_codes_only():
    assert flat_metrics({"14A1", "25F23"}, {"14A1", "11H"}) == {"precision": 0.5, "recall": 0.5, "f1": 0.5}
    assert flat_metrics(set(), {"14A1"}) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_closure_adds_every_ancestor_and_splits_combined_codes():
    assert ancestor_closure(["14A1"]) == {"1", "14", "14A", "14A1"}
    assert ancestor_closure(["61B:14A"]) == {"6", "61", "61B", "1", "14", "14A"}


def test_hierarchical_metrics_give_a_parent_partial_credit():
    result = hierarchical_metrics(["14A"], ["14A1"])
    assert result["h_precision"] == 1.0
    assert result["h_recall"] == 0.75
    assert result["h_f1"] == pytest.approx(6 / 7)
    assert result["h_jaccard"] == 0.75


def test_ranking_metrics_judge_the_order():
    result = ranking_metrics(["11H", "14A1", "25F", "31A"], ["14A1", "31A"], cutoffs=(1, 2))
    assert (result["precision@1"], result["hit@1"]) == (0.0, 0.0)
    assert (result["precision@2"], result["recall@2"], result["hit@2"]) == (0.5, 0.5, 1.0)
    assert result["reciprocal_rank"] == 0.5
    assert result["average_precision"] == pytest.approx((1 / 2 + 2 / 4) / 2)


def test_cutoff_applies_the_threshold_and_the_cap():
    best_first = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    assert cutoff(best_first, top_k=None, threshold=0.75) == 2
    assert cutoff(best_first, top_k=3, threshold=None) == 3
    assert cutoff(best_first, top_k=1, threshold=0.75) == 1
    assert cutoff(best_first, top_k=None, threshold=0.95) == 0


def test_select_ranks_best_first_and_breaks_ties_by_code_order():
    scores = np.array([0.5, 0.75, 0.5, 0.25], dtype=np.float32)
    codes = ["11H", "14A1", "25F", "31A"]
    expected = [("14A1", 0.75), ("11H", 0.5), ("25F", 0.5)]
    assert select(scores, codes, SelectionSettings(top_k=3)) == expected
    assert select(scores, codes, SelectionSettings(threshold=0.5)) == expected


def test_a_selection_rule_is_required():
    with pytest.raises(ValueError):
        SelectionSettings()
