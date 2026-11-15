"""Evaluation: an experiment's predictions against the true codes of a split.

evaluate() applies the experiment's selection rule and writes results/<experiment>__<split>.json.
sweep() applies many rules to the same scores and writes results/<experiment>__<split>__sweep.csv.
"""

import csv
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from caris import data, metrics
from caris.config import Experiment, Split
from caris.index import CodeIndex, load_index
from caris.retrieval import Retriever
from caris.schemas import TagsV1, answer_type
from caris.selection import cutoff, ranking
from caris.tagging import read_run, run_path

RESULTS = Path("results")
PER_IMAGE = ("hit", "precision", "recall", "f1", "h_precision", "h_recall", "h_f1", "h_jaccard")


@dataclass
class TaggedSplit:
    """A split's true codes, and what its tag run holds for each image."""

    run: Path
    header: dict
    truth: dict[str, list[str]]
    answers: dict[str, TagsV1] = field(default_factory=dict)
    failed: list[dict] = field(default_factory=list)
    untagged: list[str] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)


def load_tagged_split(experiment: Experiment, split: Split) -> TaggedSplit:
    path = run_path(experiment.tagging, split)
    if not path.exists():
        raise FileNotFoundError(f"No tag run at {path}. Tag first: caris tag <experiment> --split {split}")
    header, lines = read_run(path)
    tagged = TaggedSplit(path, header, data.load_split(split))
    answer = answer_type(experiment.tagging.output_schema)
    for image in tagged.truth:
        line = lines.get(image)
        if line is None:
            tagged.untagged.append(image)
            continue
        tagged.seconds.append(line["seconds"])
        if line["error"]:
            tagged.failed.append({"image": image, "error": line["error"]})
        else:
            tagged.answers[image] = answer.model_validate(line["answer"])
    return tagged


def evaluate(experiment: Experiment, split: Split) -> tuple[Path, dict]:
    """Scores the experiment's selection rule on a split, and writes the results file."""
    tagged, index, retriever = _prepare(experiment, split)
    rule = experiment.selection
    per_image = []
    for image, answer in tagged.answers.items():
        scores = retriever.scores(answer.queries())
        order = ranking(scores)
        ranked = [index.codes[i] for i in order]
        keep = cutoff(scores[order], rule.top_k, rule.threshold)
        truth = tagged.truth[image]
        per_image.append({
            "image": image,
            "tags": answer.tags,
            "predicted": [[index.codes[i], round(float(scores[i]), 4)] for i in order[:keep]],
            "truth": truth,
            "metrics": image_metrics(set(ranked[:keep]), set(truth)),
            "ranking": metrics.ranking_metrics(ranked, truth),
        })
    results = {
        "experiment": experiment.model_dump(by_alias=True),
        "split": split,
        "tag_run": {"file": tagged.run.name, **tagged.header},
        "index": index.meta,
        "images": {"evaluated": len(per_image), "failed": tagged.failed, "untagged": tagged.untagged},
        "summary": summarise(per_image, set(index.codes), tagged.seconds),
        "per_image": per_image,
    }
    path = RESULTS / f"{experiment.name}__{split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    return path, results


def sweep(experiment: Experiment, split: Split, thresholds: list, top_ks: list) -> Path:
    """Every pair of top_k and threshold, over the same scores. None leaves that part of the
    rule out, so one table holds the threshold curve, the top-k curve and the capped curves."""
    tagged, index, retriever = _prepare(experiment, split)
    if not tagged.answers:
        raise RuntimeError(f"{tagged.run.name} has no tagged images to sweep.")
    rules = [(k, t) for k in top_ks for t in thresholds if k is not None or t is not None]
    totals = {rule: Totals() for rule in rules}
    for image, answer in tagged.answers.items():
        scores = retriever.scores(answer.queries())
        order = ranking(scores)
        best_first = scores[order]
        truth = set(tagged.truth[image])
        for (top_k, threshold), total in totals.items():
            total.add({index.codes[i] for i in order[:cutoff(best_first, top_k, threshold)]}, truth)

    rows = [{"top_k": top_k, "threshold": threshold, **total.row()} for (top_k, threshold), total in totals.items()]
    path = RESULTS / f"{experiment.name}__{split}__sweep.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def paired_difference(first: dict, second: dict, metric: str) -> dict:
    """How much the second results file beats the first on a per-image metric, over the images
    both evaluated, with a 95% bootstrap interval. If the interval spans 0, the difference isn't clear."""
    a = {entry["image"]: entry["metrics"][metric] for entry in first["per_image"]}
    b = {entry["image"]: entry["metrics"][metric] for entry in second["per_image"]}
    differences = [b[image] - a[image] for image in sorted(a.keys() & b.keys())]
    return {"images": len(differences), "mean": statistics.fmean(differences), "ci95": interval(differences)}


def image_metrics(predicted: set[str], truth: set[str]) -> dict:
    return {
        "hit": 1.0 if predicted & truth else 0.0,
        **metrics.flat_metrics(predicted, truth),
        **metrics.hierarchical_metrics(predicted, truth),
    }


def summarise(per_image: list[dict], indexed: set[str], seconds: list[float]) -> dict:
    """Per-image means with bootstrap intervals, pooled metrics, ranking means, the recall
    ceiling, a breakdown by top-level branch, list length and tagging time."""
    if not per_image:
        return {}
    predicted = [{code for code, _ in entry["predicted"]} for entry in per_image]
    truth = [set(entry["truth"]) for entry in per_image]
    hits = sum(len(p & t) for p, t in zip(predicted, truth))
    return {
        **{name: mean_and_interval([entry["metrics"][name] for entry in per_image]) for name in PER_IMAGE},
        "pooled": pooled(hits, sum(map(len, predicted)), sum(map(len, truth))),
        "ranking": {name: statistics.fmean(entry["ranking"][name] for entry in per_image)
                    for name in per_image[0]["ranking"]},
        "ceiling": ceiling(truth, indexed),
        "by_branch": by_branch(predicted, truth),
        "codes_per_image": statistics.fmean(map(len, predicted)),
        "median_seconds_per_image": statistics.median(seconds) if seconds else None,
    }


def mean_and_interval(values: list[float]) -> dict:
    return {"mean": statistics.fmean(values), "ci95": interval(values)}


def interval(values: list[float], draws: int = 1000, seed: int = 0) -> list[float]:
    """The 95% percentile bootstrap interval of the mean, resampling images."""
    resampled = np.random.default_rng(seed).choice(np.asarray(values, dtype=float), size=(draws, len(values)))
    return np.percentile(resampled.mean(axis=1), [2.5, 97.5]).tolist()


def pooled(hits: int, predicted: int, true: int) -> dict:
    """Precision and recall over all images' codes at once. Unlike per-image means, these stay
    meaningful when a strict threshold leaves some images without any code."""
    precision = hits / predicted if predicted else 0.0
    recall = hits / true if true else 0.0
    return {"precision": precision, "recall": recall, "f1": metrics.f1(precision, recall)}


def ceiling(truth: list[set[str]], indexed: set[str]) -> dict:
    """The best recall possible: the share of true codes that the index has at all."""
    return {
        "recall": statistics.fmean(len(t & indexed) / len(t) for t in truth),
        "pooled_recall": sum(len(t & indexed) for t in truth) / sum(map(len, truth)),
    }


def by_branch(predicted: list[set[str]], truth: list[set[str]]) -> dict:
    """Pooled precision and recall per top-level Iconclass branch, the first character of a code."""
    counts = defaultdict(Counter)
    for p, t in zip(predicted, truth):
        for kind, group in (("predicted", p), ("true", t), ("hits", p & t)):
            for code in group:
                counts[code[0]][kind] += 1
    return {
        branch: {
            "precision": c["hits"] / c["predicted"] if c["predicted"] else None,
            "recall": c["hits"] / c["true"] if c["true"] else None,
            "true_codes": c["true"],
        }
        for branch, c in sorted(counts.items())
    }


@dataclass
class Totals:
    """Running sums over the images of a sweep, for one selection rule."""

    images: int = 0
    sums: Counter = field(default_factory=Counter)
    hits: int = 0
    predicted: int = 0
    true: int = 0

    def add(self, predicted: set[str], truth: set[str]):
        self.images += 1
        self.sums.update(image_metrics(predicted, truth))
        self.hits += len(predicted & truth)
        self.predicted += len(predicted)
        self.true += len(truth)

    def row(self) -> dict:
        means = {name: round(self.sums[name] / self.images, 4) for name in PER_IMAGE}
        overall = pooled(self.hits, self.predicted, self.true)
        return {
            **means,
            **{f"pooled_{name}": round(value, 4) for name, value in overall.items()},
            "codes_per_image": round(self.predicted / self.images, 2),
        }


def _prepare(experiment: Experiment, split: Split) -> tuple[TaggedSplit, CodeIndex, Retriever]:
    """Loads a split's tags and the index, and embeds every tag text in one go."""
    tagged = load_tagged_split(experiment, split)
    index = load_index(experiment.embedding)
    retriever = Retriever(index, experiment.embedding)
    retriever.embed([text for answer in tagged.answers.values() for text, _ in answer.queries()])
    return tagged, index, retriever
