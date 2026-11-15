"""Tagging: a VLM looks at every image of a split and answers in a fixed schema.

All answers of one tagging setup on one split form a tag run: a JSONL file whose first line
is a header describing what produced it, followed by one line per image. A run picks up
where it stopped, so a session that ends early loses at most the image it was on. Any error
from Ollama stops the run; only a reply that fails validation is recorded as a failed image.
"""

import json
import time
from importlib.resources import files
from pathlib import Path

import ollama
from pydantic import ValidationError
from tqdm import tqdm

from caris import data, provenance
from caris.config import Split, TaggingSettings
from caris.schemas import json_schema, parse

RUNS = Path("runs/tags")


def prompt_text(name: str) -> str:
    return files("caris").joinpath("prompts", f"{name}.txt").read_text(encoding="utf-8")


def ask(image: Path, tagging: TaggingSettings) -> str:
    """The VLM's raw reply about one image."""
    response = ollama.chat(
        model=tagging.model,
        messages=[{"role": "user", "content": prompt_text(tagging.prompt), "images": [image]}],
        format=json_schema(tagging.output_schema),
        options={"temperature": tagging.temperature, "seed": tagging.seed},
        think=tagging.think,
    )
    return response.message.content


def run_path(tagging: TaggingSettings, split: Split) -> Path:
    """A run is named after what produced it, plus a hash of every tagging setting."""
    parts = (provenance.slug(tagging.model), tagging.prompt, tagging.output_schema, split, tagging.fingerprint())
    return RUNS / ("__".join(parts) + ".jsonl")


def read_run(path: Path) -> tuple[dict, dict[str, dict]]:
    """The run's header, and its line for each image. Every complete line ends in a newline,
    so a last line without one was cut off mid-write and is left out."""
    with path.open(encoding="utf-8", newline="") as f:
        header = _parse_line(path, 1, next(f))["header"]
        lines = [_parse_line(path, number, line) for number, line in enumerate(f, 2) if line.endswith("\n")]
        return header, {line["image"]: line for line in lines}


def _parse_line(path: Path, number: int, line: str) -> dict:
    try:
        return json.loads(line)
    except json.JSONDecodeError as error:
        problem = f"{error.msg.removesuffix(' at')} at column {error.colno}"  # json's messages may end in "at"
        raise RuntimeError(f"Line {number} of {path} is not valid JSON ({problem}).") from error


def tag_split(tagging: TaggingSettings, split: Split, limit: int | None = None) -> Path:
    """Tags every image of the split that the run doesn't have yet. Returns the run's path."""
    path = run_path(tagging, split)
    header = _header(tagging, split)
    if path.exists():
        started, done = read_run(path)
        _check_same_setup(started, header, path)
        _drop_cut_off_line(path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"header": header}) + "\n", encoding="utf-8")
        done = {}

    todo = [image for image in list(data.load_split(split))[:limit] if image not in done]
    with path.open("a", encoding="utf-8") as run:
        for image in tqdm(todo, desc=f"Tagging {split} with {tagging.model}", unit="image"):
            run.write(json.dumps(_tag(image, split, tagging), ensure_ascii=False) + "\n")
            run.flush()
    return path


def _tag(image: str, split: Split, tagging: TaggingSettings) -> dict:
    start = time.perf_counter()
    reply = ask(data.image_path(split, image), tagging)
    seconds = round(time.perf_counter() - start, 2)
    try:
        answer, error = parse(reply, tagging.output_schema).model_dump(), None
    except ValidationError as e:
        answer, error = None, f"invalid reply: {e.errors()[0]['msg']}"
    return {"image": image, "answer": answer, "reply": reply, "seconds": seconds, "error": error}


def _header(tagging: TaggingSettings, split: Split) -> dict:
    return {
        "tagging": tagging.model_dump(by_alias=True),
        "split": split,
        "model_digest": provenance.model_digest(tagging.model),
        "prompt_hash": provenance.text_hash(prompt_text(tagging.prompt)),
        **provenance.stamp(),
    }


def _drop_cut_off_line(path: Path):
    """Removes a last line cut off mid-write, so that the next line starts on a line of its own.
    read_run has already left that line out, so its image gets tagged again."""
    content = path.read_bytes()
    complete = content.rfind(b"\n") + 1
    if complete < len(content):
        with path.open("r+b") as f:
            f.truncate(complete)
        print(f"Note: the last line of {path.name} was cut off mid-write. Its image will be tagged again.")


def _check_same_setup(started: dict, now: dict, path: Path):
    """Resuming with other weights or another prompt text would mix two setups in one run."""
    changed = [key for key in ("model_digest", "prompt_hash") if started[key] != now[key]]
    if changed:
        raise RuntimeError(f"{path.name} was started with a different {' and '.join(changed)}. "
                           "Give the prompt a new name, or move the old run aside.")
    if started["ollama_version"] != now["ollama_version"]:
        print(f"Note: {path.name} was started under Ollama {started['ollama_version']}, "
              f"and this session runs {now['ollama_version']}.")
