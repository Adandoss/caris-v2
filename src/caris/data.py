"""The dataset: Iconclass's labelled test set of images, split three ways.

    test   1,000 images, tagged, used only for reported numbers
    val      400 images, tagged, used to choose thresholds and filter settings
    train  the rest, labels only: code frequencies, filter training, the recommendation collection

Images without any code are left out. The split is seeded, so running it again gives the
same images. Set CARIS_DATA to keep the data somewhere other than ./data (as on Kaggle).
"""

import json
import os
import random
import shutil
import zipfile
from pathlib import Path

DATA = Path(os.environ.get("CARIS_DATA", "data"))
DATASET_URL = "https://iconclass.org/testset/779ba2ca9e977c58d818e3823a676973.zip"
SIZES = {"test": 1000, "val": 400}  # train gets every other image
SEED = 2026


def split_dataset(zip_path: Path) -> dict[str, int]:
    """Writes data/splits/<split>.json and extracts the test and val images. Returns the split sizes."""
    if not zip_path.exists():
        raise FileNotFoundError(f"No dataset at {zip_path}. Download it from {DATASET_URL}")
    with zipfile.ZipFile(zip_path) as archive:
        labels = {image: _clean(codes) for image, codes in json.loads(archive.read("data.json")).items()}
        images = sorted(image for image, codes in labels.items() if codes)
        random.Random(SEED).shuffle(images)

        splits, start = {}, 0
        for name, size in SIZES.items():
            splits[name] = images[start:start + size]
            start += size
        splits["train"] = images[start:]

        for name, members in splits.items():
            _write_json(DATA / "splits" / f"{name}.json", {image: labels[image] for image in members})
        _extract(archive, {image: name for name in SIZES for image in splits[name]})
    return {name: len(members) for name, members in splits.items()}


def load_split(split: str) -> dict[str, list[str]]:
    """Each image of the split with its true codes, in the split's seeded order."""
    path = DATA / "splits" / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"No split at {path}. Create it with: caris split")
    return json.loads(path.read_text(encoding="utf-8"))


def image_path(split: str, image: str) -> Path:
    return DATA / "images" / split / image


def _clean(codes: list[str]) -> list[str]:
    return sorted({code.strip() for code in codes if code.strip()})


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False), encoding="utf-8")


def _extract(archive: zipfile.ZipFile, split_of: dict[str, str]):
    """Copies each wanted image out of the archive into data/images/<split>/."""
    for member in archive.namelist():
        name = Path(member).name
        if name in split_of:
            target = image_path(split_of[name], name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as copy:
                shutil.copyfileobj(source, copy)
