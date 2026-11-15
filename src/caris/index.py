"""The code index: every Iconclass code's text, embedded once per embedding setup."""

import json
from dataclasses import dataclass
from pathlib import Path

import iconclass
import numpy as np

from caris import codes, provenance
from caris.config import EmbeddingSettings
from caris.embedding import embed

INDEXES = Path("cache/indexes")


@dataclass(frozen=True)
class CodeIndex:
    codes: list[str]  # sorted, so codes with equal scores rank in code order
    vectors: np.ndarray  # one unit-length row per code
    meta: dict

    def similarities(self, queries: np.ndarray) -> np.ndarray:
        """The cosine similarity of every query row to every code, shape (queries, codes)."""
        return queries @ self.vectors.T


def index_path(embedding: EmbeddingSettings) -> Path:
    name = f"{provenance.slug(embedding.model)}__{embedding.text}"
    if embedding.document_prefix:
        name += f"__{provenance.text_hash(embedding.document_prefix)}"
    return INDEXES / f"{name}.npz"


def build_index(embedding: EmbeddingSettings) -> Path:
    code_list = sorted(codes.texts())
    documents = [codes.document_text(code, embedding.text) for code in code_list]
    vectors = embed(documents, embedding.model, embedding.document_prefix)
    meta = {
        "embedding": embedding.model_dump(),
        "model_digest": provenance.model_digest(embedding.model),
        "iconclass_version": iconclass.__version__,
        "documents_hash": provenance.text_hash("\n".join(documents)),
        **provenance.stamp(),
    }
    path = index_path(embedding)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, codes=np.array(code_list), vectors=vectors, meta=json.dumps(meta))
    return path


def load_index(embedding: EmbeddingSettings) -> CodeIndex:
    """The index for this embedding setup. Refuses one built with other model weights."""
    path = index_path(embedding)
    if not path.exists():
        raise FileNotFoundError(f"No index at {path}. Build it with: caris index <experiment>")
    with np.load(path) as stored:
        index = CodeIndex(stored["codes"].tolist(), stored["vectors"], json.loads(stored["meta"].item()))
    if index.meta["model_digest"] != provenance.model_digest(embedding.model):
        raise RuntimeError(f"{path} was built with other {embedding.model} weights. Rebuild it: caris index <experiment>")
    return index
