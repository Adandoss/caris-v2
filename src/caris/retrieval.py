"""From an answer's queries to one score per Iconclass code."""

import numpy as np

from caris.config import EmbeddingSettings
from caris.embedding import embed_queries
from caris.index import CodeIndex


class Retriever:
    """Scores every code of an index against a list of (text, weight) queries.

    A code's score is its best weighted cosine similarity to any of the queries. Each text is
    embedded only once, so scoring many images costs one embedding per distinct tag.
    """

    def __init__(self, index: CodeIndex, embedding: EmbeddingSettings):
        self.index = index
        self.embedding = embedding
        self._vectors: dict[str, np.ndarray] = {}

    def embed(self, texts: list[str]):
        """Embeds texts ahead of scoring, in as few calls as possible."""
        missing = [text for text in dict.fromkeys(texts) if text not in self._vectors]
        if missing:
            self._vectors.update(zip(missing, embed_queries(missing, self.embedding)))

    def scores(self, queries: list[tuple[str, float]]) -> np.ndarray:
        texts = [text for text, _ in queries]
        self.embed(texts)
        vectors = np.stack([self._vectors[text] for text in texts])
        weights = np.array([weight for _, weight in queries], dtype=np.float32)
        return (weights[:, None] * self.index.similarities(vectors)).max(axis=0)
