"""What produced a file: model digests, the Ollama version, hashes, the time and the place.

Tag runs and indexes record these, so that every number can be traced back to its cause.
"""

import hashlib
import json
import os
import urllib.request
from datetime import datetime, timezone

import ollama


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def slug(name: str) -> str:
    """A model name made safe for file names: gemma4:26b becomes gemma4-26b."""
    return name.replace(":", "-").replace("/", "-")


def model_digest(model: str) -> str:
    """The digest of a pulled model. It changes whenever different weights are pulled."""
    wanted = model if ":" in model else f"{model}:latest"
    for pulled in ollama.list().models:
        if pulled.model == wanted:
            return pulled.digest
    raise LookupError(f"{model} isn't pulled. Run: ollama pull {model}")


def ollama_version() -> str | None:
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    if "://" not in host:
        host = f"http://{host}"
    try:
        with urllib.request.urlopen(f"{host}/api/version", timeout=5) as response:
            return json.load(response)["version"]
    except OSError:
        return None


def stamp() -> dict:
    """When, where and under which Ollama version a file was made."""
    return {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": "kaggle" if "KAGGLE_KERNEL_RUN_TYPE" in os.environ else "local",
        "ollama_version": ollama_version(),
    }
