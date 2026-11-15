# CARIS v2

CARIS suggests Iconclass codes for an image of an artwork, and similar artworks from a labelled collection:

1. A local vision-language model (VLM), served by Ollama, tags the image.
2. The tags are matched to every Iconclass code's text by embedding similarity.
3. The best-matching codes are kept, by a top-k cap, a similarity threshold, or both.
4. Three recommenders each pick the most similar artwork from the collection.

It is also a test bench. The VLM, prompt, output schema, embedding model, code text and selection rule can each be changed and measured on their own. [ARCHITECTURE.md](ARCHITECTURE.md) explains the design and the reasons behind it.

## Setup

You need Python 3.11 or newer, and [Ollama](https://ollama.com) running.

```bash
python -m venv .venv
.venv\Scripts\activate            # on macOS or Linux: source .venv/bin/activate
pip install -e ".[dev,analysis]"
ollama pull gemma4:26b
ollama pull mxbai-embed-large
```

Download the Iconclass test set (3.3 GB) into `data/`, then split it once:

```bash
mkdir data
curl -L -o data/iconclass_dataset.zip https://iconclass.org/testset/779ba2ca9e977c58d818e3823a676973.zip
caris split
```

`caris split` writes `data/splits/{train,val,test}.json` and extracts the 1,400 val and test images. The split is seeded, so it always comes out the same.

## Running an experiment

An experiment is a TOML file in `experiments/` that names the split, the tagging setup, the embedding setup and the selection rule.

```bash
caris index experiments/gemma4-26b__mxbai__standalone.toml               # embed every code text, once per embedding setup
caris tag experiments/gemma4-26b__mxbai__standalone.toml --split val     # the VLM step; resumes where it stopped
caris sweep experiments/gemma4-26b__mxbai__standalone.toml --split val   # precision and recall for many selection rules
caris tag experiments/gemma4-26b__mxbai__standalone.toml                 # the experiment's own split, test
caris evaluate experiments/gemma4-26b__mxbai__standalone.toml            # writes results/<experiment>__test.json
```

- Choose the threshold on `val`, then report on `test`.
- To try a setup quickly, add `--limit 20` to `caris tag`.
- Experiments with the same `[tagging]` section share a tag run, and experiments with the same `[embedding]` section share an index. So comparing two code texts or two thresholds costs no VLM time.

To get the codes and similar artworks for one image:

```bash
caris predict painting.jpg --experiment experiments/gemma4-26b__mxbai__standalone.toml
```

### Sweeps

`caris sweep` tries every pair of a threshold and a top-k cap. The defaults are:

- thresholds: none, then 0.55 to 0.95 in steps of 0.01 (`--thresholds none,0.55:0.95:0.01`);
- caps: none, 1, 2, 5, 10, 20 and 50 (`--top-k none,1,2,5,10,20,50`).

So one table holds the top-k curve, the threshold curve and the capped threshold curves. `notebooks/analysis.ipynb` plots them, and puts the results files side by side.

## Tagging on Kaggle

`notebooks/kaggle_tag.ipynb` runs `caris tag` on a free Kaggle GPU. Nothing else needs a GPU.

1. Upload `data/splits/` and `data/images/` as a private Kaggle dataset.
2. Follow the steps at the top of the notebook.
3. Tagging stops after `TAGGING_HOURS`, so the version keeps its output. If a tag run isn't done by then, the notebook says so, and its top explains how a new version carries on from that output.
4. Copy the finished tag runs into `runs/tags/` here.

If the data lives anywhere other than `./data`, set `CARIS_DATA` to its folder.

## Files

| Path | Holds | In git |
|---|---|---|
| `experiments/` | experiment files | yes |
| `runs/tags/` | tag runs: one JSONL file per tagging setup and split | yes |
| `results/` | results files and sweeps | yes |
| `cache/` | code indexes, tag embeddings, the Iconclass texts | no, rebuilt when needed |
| `data/` | the dataset zip, the split, the val and test images | no |

## Tests

```bash
pytest
```
