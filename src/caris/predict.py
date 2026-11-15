"""One image, end to end: its tags, its codes with their texts, and each recommender's pick."""

from dataclasses import dataclass
from pathlib import Path

from caris import codes, data
from caris.config import Experiment
from caris.index import load_index
from caris.recommend import recommend
from caris.retrieval import Retriever
from caris.schemas import parse
from caris.selection import select
from caris.tagging import ask


@dataclass
class Prediction:
    tags: list[str]
    codes: list[tuple[str, float, str]]  # code, score, text
    recommendations: dict[str, str | None]  # recommender -> an artwork of the train split


def predict(image: Path, experiment: Experiment) -> Prediction:
    if not image.exists():
        raise FileNotFoundError(f"No image at {image}")
    answer = parse(ask(image, experiment.tagging), experiment.tagging.output_schema)
    index = load_index(experiment.embedding)
    scores = Retriever(index, experiment.embedding).scores(answer.queries())
    selected = select(scores, index.codes, experiment.selection)
    picks = recommend(data.load_split("train"), [code for code, _ in selected], experiment.recommendation.idf_impact)
    return Prediction(answer.tags, [(code, score, codes.texts()[code]) for code, score in selected], picks)
