"""Text embeddings from Ollama, one unit-length row per text, and a cache for tag texts."""

from pathlib import Path

import numpy as np
import ollama
from tqdm import tqdm

from caris import provenance
from caris.config import EmbeddingSettings

QUERIES = Path("cache/queries")
BATCH_SIZE = 128


def embed(texts: list[str], model: str, prefix: str = "") -> np.ndarray:
    rows = []
    batches = range(0, len(texts), BATCH_SIZE)
    for start in tqdm(batches, desc=f"Embedding with {model}", disable=len(batches) < 2):
        batch = [prefix + text for text in texts[start:start + BATCH_SIZE]]
        rows.extend(ollama.embed(model=model, input=batch).embeddings)
    vectors = np.asarray(rows, dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def embed_queries(texts: list[str], embedding: EmbeddingSettings) -> np.ndarray:
    """Embeds tag texts, reusing every vector cached for the same model weights and prefix."""
    digest = provenance.model_digest(embedding.model)[:12]
    prefix = provenance.text_hash(embedding.query_prefix)
    path = QUERIES / f"{provenance.slug(embedding.model)}__{digest}__{prefix}.npz"
    cached = _load(path)
    missing = sorted(set(texts) - cached.keys())
    if missing:
        cached.update(zip(missing, embed(missing, embedding.model, embedding.query_prefix)))
        _save(path, cached)
    return np.stack([cached[text] for text in texts])


def _load(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path) as stored:
        return dict(zip(stored["texts"].tolist(), stored["vectors"]))


def _save(path: Path, vectors: dict[str, np.ndarray]):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, texts=np.array(list(vectors)), vectors=np.stack(list(vectors.values())))
