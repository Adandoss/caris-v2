"""Experiment settings.

An experiment is a TOML file holding everything that produces one row of a results table:
which split, how images are tagged, how codes are embedded and how codes are selected.
A misspelled or mistyped setting is an error, never a silent default.
"""

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Split = Literal["val", "test"]


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def fingerprint(self) -> str:
        """A short hash of every setting in the section, used to name the files it produces."""
        settings = json.dumps(self.model_dump(by_alias=True), sort_keys=True)
        return hashlib.sha256(settings.encode()).hexdigest()[:8]


class DataSettings(Section):
    split: Split = "test"


class TaggingSettings(Section):
    model: str = "gemma4:26b"
    prompt: str = "tags_v1"  # src/caris/prompts/<prompt>.txt
    # "none" enforces nothing and reads the reply as a comma-separated list, the old way.
    output_schema: Literal["tags_v1", "none"] = Field("tags_v1", alias="schema")
    temperature: float = Field(0.0, ge=0)
    seed: int = 42
    # Gemma 4, Qwen 3.5 and Ornith 1.5 can reason before answering, and Ollama turns that
    # on by default. It costs far more time, so it is an explicit choice here.
    think: bool = False


class EmbeddingSettings(Section):
    model: str = "mxbai-embed-large"
    text: Literal["standalone", "parent", "hierarchical"] = "standalone"
    query_prefix: str = ""
    document_prefix: str = ""


class SelectionSettings(Section):
    """Keep the top_k best codes, the codes scoring at least threshold, or both."""

    top_k: int | None = Field(None, gt=0)
    threshold: float | None = None

    @model_validator(mode="after")
    def _has_a_rule(self):
        if self.top_k is None and self.threshold is None:
            raise ValueError("set top_k, threshold or both")
        return self


class RecommendationSettings(Section):
    idf_impact: float = 1.0  # the exponent on IDF weights in the IDF recommender


class Experiment(Section):
    name: str
    data: DataSettings = DataSettings()
    tagging: TaggingSettings = TaggingSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    selection: SelectionSettings = SelectionSettings(top_k=20)
    recommendation: RecommendationSettings = RecommendationSettings()


def load_experiment(path: Path) -> Experiment:
    """Reads an experiment file. The experiment is named after the file."""
    with path.open("rb") as f:
        try:
            settings = tomllib.load(f)
        except tomllib.TOMLDecodeError as error:
            raise RuntimeError(f"{path} is not valid TOML: {error}") from error
    return Experiment.model_validate({**settings, "name": path.stem})
