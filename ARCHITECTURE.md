# CARIS: slim architecture

> **Status:** built 15 November 2026 in this repository. The previous system stays, unchanged, in its own repository.
>
> Numbers given as reasons for decisions come from runs of the previous system on its own 1,009-image eval split. None of that system's data is carried over (see D10).

## 1. What the system does

CARIS takes an image of an artwork and returns the Iconclass codes that describe it, together with similar artworks from a labelled collection:

1. A local vision-language model (VLM) tags the image.
2. The tags are embedded and compared with embeddings of every Iconclass code's text.
3. The closest codes are selected.
4. Three recommenders each use the selected codes to pick the most similar artwork from the collection.

The same code base is also a test bench. Each stage can be varied on its own (VLM, prompt, output schema, embedding model, code text, selection rule, filter) and measured against ground-truth codes.

## 2. Scope

### Carried over from the previous system

| Part | What happened to it |
|---|---|
| Flat, hierarchical and ranking metrics | Definitions ported unchanged into `metrics.py` |
| Seeded, order-preserving dataset split | Extended to train / val / test (D8) |
| Download of the English Iconclass texts | Moved into `codes.py`, downloaded fresh |
| The Gemma keyword prompt | Kept as `keywords_v0`; its JSON version is `tags_v1` |
| Evaluation integrity rules | Failed and untagged images are listed and left out of the metrics; any Ollama error stops a tag run |
| The three recommenders: prefix similarity, IDF and Jaccard | Ported into `recommend.py`, with two fixes to how codes are compared (D9) |

### Left behind

| Part | Reason |
|---|---|
| Structural (keyword) mapping and its keyword index | D2 |
| YOLO, RAM++, the detector registry, the COCO evaluation | D3 |
| ChromaDB | D4 |
| The large test suite | D13 |
| The bibliasacra.nl scraper and `scraped_data/` | Nothing read them; the 30,806 scraped entries had no codes and no images |
| `meanings.py`, most of `data_io.py`, `timing.py`, `runner.py`, `stats.py`, `ranking.py` | Never called, or folded into their only caller |
| `Makefile`, `caris.example.toml`, `improvements.md` | Replaced by the CLI, experiment files and this document |
| ultralytics, transformers, einops, timm, chromadb, requests, torch, pillow, beautifulsoup4 | Only the parts left behind needed them |

## 3. Architecture

### 3.1 Stages and data flow

```
                        needs a model · output cached as a file
  images ──── tag (VLM) ─────────► runs/tags/<tag-run>.jsonl                GPU, local or Kaggle
                                             │
  Iconclass ── index (embedder) ─► cache/indexes/<embedder>__<text>.npz     once per variant
                                             │
                        no VLM calls · seconds
                                             ▼
              retrieve ──► select ──► evaluate ──► results/<experiment>__<split>.json
                                 └──► sweep ─────► results/<experiment>__<split>__sweep.csv

  one image ── predict: tag → retrieve → select → recommend ──► codes, texts, similar artworks
```

| Stage | Reads | Writes | Cost |
|---|---|---|---|
| `split` | the dataset zip | `data/splits/{train,val,test}.json`, val and test images | once, a few seconds |
| `index` | Iconclass texts and tree, an embedding model | one `.npz` per (embedding model, text variant) | once per variant |
| `tag` | images, a VLM, a prompt, a schema | one tag run per (tagging setup, split) | ~23 s per image for Gemma 4 26B locally |
| `retrieve` | a tag run, an index | one score per code per image, in memory | seconds once the tags are embedded |
| `select` | scores, a selection rule | predicted codes per image | milliseconds |
| `evaluate`, `sweep` | predictions, ground truth | a results file, a sweep table | seconds to minutes |
| `predict` | one image | codes with texts, and the best match from each recommender | one VLM call |

### 3.2 Project layout

```
CARIS_v2/
├── pyproject.toml            runtime: iconclass, numpy, ollama, pydantic, tqdm
├── README.md                 setup and commands
├── ARCHITECTURE.md           this document
├── experiments/*.toml        one file per reported condition
├── notebooks/
│   ├── kaggle_tag.ipynb      installs Ollama, pulls the model, runs `caris tag`
│   └── analysis.ipynb        tables and plots from results/
├── tests/test_metrics.py     hand-made checks of the metrics and the selection (D13)
├── runs/tags/                tag runs (committed: they are research data)
├── results/                  results files and sweeps (committed)
├── cache/                    indexes, query embeddings, Iconclass texts (ignored, rebuildable)
├── data/                     dataset zip, splits, val and test images (ignored)
└── src/caris/
    ├── config.py             experiment settings (pydantic) and TOML loading
    ├── provenance.py         model digests, Ollama version, hashes, time and machine
    ├── codes.py              Iconclass texts and tree: lineage(), document_text(code, variant)
    ├── data.py               the three-way split; extracts only the images that get tagged
    ├── prompts/              tags_v1.txt (JSON), keywords_v0.txt (comma list, for the ablation)
    ├── schemas.py            VLM output schemas; each answer turns into (text, weight) queries
    ├── tagging.py            image → validated answer via Ollama; resumable tag runs
    ├── embedding.py          Ollama embeddings with per-model prefixes; query cache
    ├── index.py              build, save and load the code index
    ├── retrieval.py          queries → one score per code for an image
    ├── selection.py          top-k, threshold or both; later the filter model
    ├── metrics.py            flat, hierarchical and ranking metrics
    ├── evaluate.py           results files, sweeps, bootstrap intervals, breakdowns
    ├── recommend.py          three recommenders over the train labels: prefix, IDF, Jaccard
    ├── predict.py            one image, end to end
    └── cli.py                caris split | index | tag | sweep | evaluate | predict
```

The prompts sit inside the package, so an installed copy, such as the one on Kaggle, always has them. `runs/` and `results/` appear with the first run.

### 3.3 Files between stages

**Tag run:** `runs/tags/<model>__<prompt>__<schema>__<split>__<hash>.jsonl`. The hash covers every tagging setting. The first line is a header, then one line per image:

```jsonc
{"header": {"tagging": {"model": "gemma4:26b", "prompt": "tags_v1", "schema": "tags_v1",
                        "temperature": 0.0, "seed": 42, "think": false},
            "split": "test", "model_digest": "…", "prompt_hash": "…",
            "created": "…", "machine": "local", "ollama_version": "…"}}
{"image": "….jpg", "answer": {"tags": ["…", "…"]}, "reply": "…", "seconds": 21.4, "error": null}
{"image": "….jpg", "answer": null, "reply": "…", "seconds": 19.8, "error": "invalid reply: …"}
```

- `reply` is the model's raw text, and `answer` is that text validated against the schema.
- Running `caris tag` again with the same experiment resumes the run. It first checks that the model digest and the prompt text haven't changed, then tags only the images that have no line yet.
- Every complete line ends in a newline. A last line without one was cut off mid-write when a session ended. Reading the run leaves it out, and resuming removes it and tags its image again. Any other line that isn't valid JSON is an error that names the line.
- Any error from Ollama stops the run. Only a reply that fails validation is recorded as a failed image.

**Index:** `cache/indexes/<embedding-model>__<text-variant>.npz` holds:
- `codes`: the code ids, sorted,
- `vectors`: one L2-normalised float32 row per code,
- `meta`: the embedding settings, the model digest, the `iconclass` version, a hash of the embedded documents, and when, where and under which Ollama version it was built.

Loading an index fails if the embedding model's weights have changed since it was built.

**Query cache:** `cache/queries/<embedding-model>__<digest>__<prefix-hash>.npz` maps tag text to its vector. All tag runs share it, so evaluating again never re-embeds a tag.

**Results:** `results/<experiment>__<split>.json` holds:
- the experiment, the split, the tag-run header and the index metadata;
- the evaluated, failed and untagged images;
- the summary:
  - `hit`, `precision`, `recall`, `f1`, `h_precision`, `h_recall`, `h_f1` and `h_jaccard`, each as a per-image mean with a 95% bootstrap interval;
  - `pooled` precision, recall and F1 over all codes at once;
  - the ranking-metric means;
  - the recall `ceiling`, a per-branch breakdown (`by_branch`), `codes_per_image` and `median_seconds_per_image`;
- one entry per image: tags, predicted codes with scores, true codes, metrics and ranking metrics.

**Sweep:** `results/<experiment>__<split>__sweep.csv` has one row per (top-k cap, threshold), with the per-image means, the pooled precision, recall and F1, and the mean list length.

### 3.4 Experiment files

An experiment is everything needed to produce one row of a results table:

```toml
# experiments/gemma4-26b__mxbai__hierarchical.toml
[data]
split = "test"                  # test | val

[tagging]
model = "gemma4:26b"
prompt = "tags_v1"              # src/caris/prompts/tags_v1.txt
schema = "tags_v1"              # "none": prompt-only comma list (the ablation)
temperature = 0.0
seed = 42
think = false                   # reasoning before answering; see D3

[embedding]
model = "mxbai-embed-large"
text = "hierarchical"           # standalone | hierarchical
query_prefix = "Represent this sentence for searching relevant passages: "

[selection]
top_k = 20                      # and/or: threshold = 0.75, chosen on val with caris sweep

[recommendation]                # optional; only caris predict reads it
idf_impact = 1.0                # exponent on the IDF weights in the IDF recommender
```

- An unknown key or a wrong type is an error, never a silent default.
- `[selection]` needs `top_k`, `threshold` or both. Leaving out the whole section means `top_k = 20`.
- Experiments with the same `[tagging]` section and split share a tag run. Experiments with the same `[embedding]` section share an index.
- The file name is the experiment's name in results.

### 3.5 Commands

```bash
caris split [--zip data/iconclass_dataset.zip]   # once: splits, val and test images
caris index    experiments/x.toml                # builds the index that [embedding] needs
caris tag      experiments/x.toml --split val    # GPU; resumable; --limit 20 for a quick check
caris tag      experiments/x.toml                # the experiment's own split (test)
caris sweep    experiments/x.toml --split val    # precision-recall table, no VLM calls
caris evaluate experiments/x.toml                # results/x__test.json
caris predict  painting.jpg --experiment experiments/x.toml
```

`caris sweep` tries every pair from `--thresholds` and `--top-k`. Both take comma-separated lists in which `START:STOP:STEP` is a range and `none` leaves that part of the rule out. A range never goes past `STOP`, and includes it when the steps land on it. The defaults are `none,0.55:0.95:0.01` and `none,1,2,5,10,20,50`, so one table holds three kinds of rule: top-k only, a threshold only, and a threshold with a cap.

## 4. Key decisions

| # | Decision | Main reason |
|---|---|---|
| D1 | Split the pipeline at the tags | The VLM takes ~23 s per image; everything after it takes milliseconds |
| D2 | Semantic retrieval only | Structural mapping: 20% hit rate with 1,507 codes per image, against 52% with 80 |
| D3 | Local VLMs through Ollama only | The study is about local VLMs; one backend works locally and on Kaggle |
| D4 | NumPy index instead of ChromaDB | Thresholds need exact scores for every code |
| D5 | Per-image top-k, threshold, or both | A per-tag k gave 1,222 codes per image |
| D6 | Structured, versioned VLM output | Validated replies, and room for richer labels |
| D7 | Experiments as files, keyed caches, recorded provenance | Compare by changing one section; explain every number |
| D8 | Choose on val, report on test | Fair operating points and comparable curves |
| D9 | Three recommenders, ported with two fixes | Three different ideas of similarity, kept from the previous system |
| D10 | Fresh start in a new repository | Every number comes from a run with full provenance |
| D11 | Kaggle runs only the tagging stage | Free GPU hours go to the only stage that needs them |
| D12 | Small dependency set | Five runtime dependencies |
| D13 | A small test file instead of a test suite | Checks the numbers that end up in the thesis, and nothing else |

### D1. The pipeline is split at the tags

**Decision.** Tagging is a stage of its own, and its output, a *tag run*, is a file. Retrieval, selection, evaluation and recommendation read tag runs and never call the VLM.

**Why.**
- Cost is lopsided. The VLM takes ~23 s per image (Gemma 4 26B, median, previous system), while everything after it takes milliseconds. Most planned experiments (thresholds, embedding models, code texts, filters) change only the cheap part.
- Comparisons need identical tags, and VLM output isn't reliably repeatable. Two August runs of the previous system produced identical tags. A September run with the same model overlapped them at a mean Jaccard of only 0.31, and no file records what changed.
- Kaggle sessions end after 9–12 h, so the expensive stage has to resume where it stopped.

**Consequences.** Tag runs are research data. They are committed, named after the setup that produced them, and never regenerated silently. `predict` still chains every stage for a single image.

### D2. Codes are found by semantic retrieval only

**Decision.** Tags and code texts are embedded with the same model, and codes are ranked by cosine similarity. Structural mapping (exact or subset matches between tags and Iconclass keywords) is left behind.

**Why.** With Gemma 4 26B tags on the previous eval split:
- structural mapping found at least one correct code (the *hit rate*) in 20% of images, while returning 1,507 codes per image (100-image run);
- semantic retrieval with 5 codes per tag reached a 52% hit rate with 80 codes per image.

Keyword matching only works when the VLM happens to produce Iconclass's exact keyword strings.

**Consequences.** The `iconclass` package stays only for the code tree (ancestors), which hierarchical code texts and hierarchical metrics need.

### D3. Only local VLMs, through one backend

**Decision.** One tagging module sends the image and the prompt to Ollama, and the model is a config value. There are no detector classes, no registry, no YOLO and no RAM++.

**Why.**
- The research question is about local VLMs. YOLO knows 80 everyday object classes and produced 0.8 tags per artwork. RAM++ reached a 26% hit rate against Gemma's 52% at the same setting. Neither is a language model.
- Ollama runs the same way on a laptop and on Kaggle. It serves both VLMs and embedding models, and it enforces JSON-schema output for vision models (D6).

**Consequences.**
- Comparing VLMs means changing a model name.
- Gemma 4, Qwen 3.5 and Ornith 1.5 can reason before they answer, and Ollama turns that on by default for all three. Reasoning costs far more time and may change the tags. So it is an explicit setting, `think`, off by default and recorded in every tag run. A run with reasoning on is a separate experiment.
- A second backend (for example vLLM, for faster batched tagging) can be added behind the same `ask()` function if throughput ever forces it. It isn't built now.

### D4. An exact NumPy index replaces ChromaDB

**Decision.** An index is one `.npz` file holding the code ids, an L2-normalised float32 matrix of their embeddings, and metadata. The similarity to every code is one matrix product. There is one index per (embedding model, code-text variant).

**Why.**
- 40,712 codes × 1,024 dimensions (mxbai-embed-large) is about 167 MB in float32, so an exact search over every code takes milliseconds per image.
- Thresholds (D5) need the similarity to *every* code. ChromaDB returns only the k nearest, and approximately (HNSW).
- One file per variant is easy to compare, cache and copy.
- ChromaDB brings a large dependency tree (onnxruntime, grpcio, kubernetes, OpenTelemetry). The previous code also turned its squared-L2 distances into cosine by assuming unit-length vectors; explicit normalisation removes that assumption.

**Consequences.**
- Memory grows with the embedding dimension: a 4,096-dimension model would need about 667 MB, and float16 halves that.
- The embedded documents aren't stored, only their hash; `codes.document_text()` rebuilds any of them.
- An index built with other weights of the same model is refused.

### D5. Codes are selected per image: top-k, threshold, or both

**Decision.** Retrieval gives each code one score per image: its highest cosine similarity to any of the image's tags. Selection then keeps one of:
- the k best codes,
- the codes scoring at least τ,
- the k best among the codes scoring at least τ.

Equal scores rank in code order, so the selection is reproducible.

**Why.**
- The previous `k` counted per tag. 100 codes per tag over ~18 tags gave 1,222 codes per image at 0.16% precision, and 5 per tag still gave ~80. Neither is a usable answer, and a per-tag k can't express confidence.
- A threshold on the image's scores is core goal 1. The scores come from cached files, so a whole precision-recall sweep is a computation, not a rerun.
- With a threshold alone, images with more tags get longer lists, because each extra tag is another chance to pass τ. The k cap bounds that.

**Consequences.** Each query carries a weight, 1 for every tag of `TagsV1`. That is where salience will enter (an additional goal). Other ways of combining a code's similarities across tags (the mean vector, the sum of the best m) can be added in `retrieval.py`. The filter model (core goal 7) sits between retrieval and selection.

### D6. VLM output is structured and versioned

**Decision.** Each output schema is a pydantic model, starting with `TagsV1 = {"tags": [str]}`, and is passed to Ollama as a JSON schema (`format=`).
- Every reply is validated: tags are stripped and de-duplicated, and an answer with no tags is invalid.
- A reply that fails validation makes its image a failed image: it is recorded and left out of the metrics, never counted as "no tags".
- Each schema version defines how it turns into retrieval queries: a list of (text, weight) pairs.

**Why.**
- Enforced structure is core goal 6. The previous parser split the reply on commas. A malformed reply became junk tags, and an empty one became a real prediction with zero codes.
- Several additional goals are new schema versions: pre-iconographical vs iconographical labels, salience from 0 to 1, and relations between objects. Older tag runs stay readable because every run records its schema.

**Consequences.** The prompt describes the fields the schema enforces. Constrained decoding can change what a model says (Tam et al., 2024). The prompt-only mode therefore stays available as an ablation (`schema = "none"` with prompt `keywords_v0`, see `experiments/gemma4-26b__prompt-only__mxbai__standalone.toml`). It measures what enforcement itself changes.

### D7. Experiments are files, caches are keyed by what made them, provenance is recorded

**Decision.** An experiment is a TOML file with four sections, `[data]`, `[tagging]`, `[embedding]` and `[selection]`, plus an optional `[recommendation]` section that only `caris predict` reads. pydantic validates it, so an unknown key is an error.
- A tag run's name comes from `[tagging]` plus the split.
- An index's name comes from `[embedding]`.
- Every results file includes the experiment, the tag-run header and the index metadata.

**Why.**
- Experiments that differ in one section reuse everything else. Ten selection rules over one tag run cost no GPU time, and three VLMs share one index.
- The previous results files recorded no settings, which is why the tag change between August and September can't be explained. A tag-run header now records:
  - the tagging settings,
  - the model digest and the Ollama version,
  - a hash of the prompt text,
  - the split, the date and the machine (`local` or `kaggle`).

**Consequences.** Temperature 0 and a fixed seed make tags more repeatable but don't guarantee it across machines and Ollama versions. That is the other reason tag runs are frozen files.

### D8. Evaluation separates choosing from reporting

**Decision.**
- **Splits.** Three, made once by `caris split` with seed 2026. The 738 images without any code are left out.
  - `test`: 1,000 images, tagged, used only for reported numbers;
  - `val`: 400 images, tagged, used to choose thresholds and filter settings;
  - `train`: the other 85,606 images, labels only. They supply code frequencies, filter training data and the recommendation collection.
- **Metrics.**
  - flat precision, recall and F1 on exact codes;
  - hierarchical hP, hR and hF1, where each code is expanded with its ancestors (Kiritchenko et al., 2006);
  - ranking metrics (P@k, R@k, hit@k, MRR, AP) over each image's full ranking, which judge the score order independently of any threshold.
- **Averaging.** Per-image means come with 95% bootstrap intervals. Pooled precision and recall count all codes at once, and stay meaningful when a strict threshold leaves some images with no codes (whose per-image precision is 0).
- **Comparisons.** Systems are compared by whole curves, or at operating points chosen on `val`, with paired bootstrap intervals over the images both evaluated (`evaluate.paired_difference`). So the two results files must be on the same split.
- **Summaries.** Every results file also reports the recall ceiling and a breakdown by top-level Iconclass branch.

**Why.**
- Choosing τ or a filter on the images you report on inflates the results.
- Cosine scales differ between embedding models and code-text variants, so one τ means something different for each. Curves don't have that problem.
- On about 1,000 images, a single hit rate has a 95% interval of roughly ±3 points.
- **The recall ceiling.** On the previous eval split, 26% of ground-truth codes didn't exist in the index as written, so retrieval could never return them:
  - about half were key suffixes such as `56D1(+113)`,
  - a quarter were colon combinations,
  - a quarter were other bracketed text, such as mottoes (`86(SPES MEA DEUS)`) and names.

  48% of images contained at least one such code. Hierarchical metrics still give them partial credit.
- **The skew.** On the previous split, 72% of ground-truth codes sat in branches 2–4 (nature, human being, society), and only 9% in 7–9 (Bible, literature, classical mythology). Gains on iconographic subjects barely move an overall mean.

**Consequences.** Each VLM is tagged on both `val` and `test`, which adds 40% to its GPU cost.

### D9. Three recommenders, ported with two fixes

**Decision.** The previous system's three recommenders stay, with the same scoring and the same output. Each one compares the predicted codes with the codes of every artwork in the collection, which is the `train` labels. Each first drops the bracketed text of a code, so `25F23(LION)` becomes `25F23` and `11H(PAUL)6` becomes `11H6`, and then counts each distinct predicted code once.

| Recommender | How an artwork scores |
|---|---|
| Prefix similarity | Each predicted code takes its best match among the artwork's codes, and the matches are summed. The same code scores 1. Codes that match once the last character of one or both is dropped (parent, child or sibling) score 0.5, and 0.25 once two characters are dropped. |
| IDF | Each predicted code that the artwork has adds its IDF raised to `idf_impact` (default 1). IDF is ln((N + 1) / (df + 1)) + 1, where N is the number of artworks in the collection and df the number that have the code. |
| Jaccard | The number of shared codes divided by the number of codes in either set. |

Each recommender returns its best-scoring artwork, or none when no artwork shares anything with the query. `caris predict` shows all three.

**Why.** Each one captures a different idea of similarity:
- prefix similarity gives partial credit for related codes;
- IDF gives more weight to rare codes;
- Jaccard measures overlap relative to the size of both code sets.

Keeping all three keeps that choice open.

**Consequences.**
- `recommend.py` keeps the previous scoring formulas. Three things changed:
  - The previous code cut each code at its first bracket, so `11H(PAUL)6`, the martyrdom of Paul, became `11H`, any male saint. That hit 14,783 of the 40,712 indexed codes, and 5,902 of them became plain `11H`. Now only the bracketed text goes.
  - The previous prefix and IDF recommenders counted a code as often as it occurred after that cut, so several saints' codes added `11H` several times. Now each distinct code counts once, as it already did in Jaccard.
  - The collection is passed in, loaded once, instead of being read from a file on every call. That doesn't change any result.
- Recommendation isn't among the research goals, so the three recommenders aren't evaluated for now. A later evaluation could check how well each one's pick matches the query's true codes.

### D10. Fresh start in a new repository

**Decision.** CARIS v2 is a new repository, and it starts with no data products:
- no tags, results, indexes or caches are imported from the previous system;
- the Iconclass texts are downloaded fresh, giving 40,712 codes;
- only the raw dataset zip is read, by `caris split`.

Code is ported only where it is correct and needed (§2). The previous repository stays as it is.

**Why.** The previous runs recorded no settings (D7), so their tags can be neither trusted as a baseline nor reproduced. Starting clean means every reported number comes from a run with full provenance.

**Consequences.**
- The split was drawn with a new seed, so the new test images haven't been looked at during earlier prompt work.
- The numbers in this document come from the previous system. They only explain decisions and won't be compared with new results.

### D11. Kaggle runs only the tagging stage

**Decision.** `notebooks/kaggle_tag.ipynb`:
- installs Ollama, pinned to the version used locally, and pulls the VLM;
- clones and installs this repository;
- runs `caris tag` on the `val` and `test` images, uploaded once as a private Kaggle dataset that `CARIS_DATA` points to;
- stops tagging `TAGGING_HOURS` after the start, so the version ends, and keeps its output, before Kaggle ends the session;
- keeps the tag runs as notebook output;
- carries on from an earlier version, when `PREVIOUS` points to that version's output added as an input.

Index building, retrieval, evaluation and recommendation run locally.

**Why.** Kaggle gives about 30 free GPU hours a week (2×T4 or a P100), in sessions of 9–12 h that run unattended once a version is committed. D1 means only the VLM calls need it. The local GPU, an RTX 4060, has 8 GB, so Gemma 4 26B (17 GB) runs mostly on the CPU there, which explains its ~23 s per image. Two T4s hold 32 GB, so the same model fits in GPU memory on Kaggle and may well run faster.

**Consequences.**
- At the local speed of ~23 s per image, one VLM costs about 6.4 h on `test` plus 2.6 h on `val`, so roughly three VLMs would fit in a weekly quota. On Kaggle it may cost less.
- Tag runs are append-only. Each version starts with an empty working folder, so a run that one version didn't finish carries on only when the next version copies it from the earlier output.
- Time 20 images on Kaggle before committing to a full run.

### D12. Small dependency set

**Decision.** Runtime dependencies are `iconclass`, `numpy`, `ollama`, `pydantic` and `tqdm`. There are two optional groups: `analysis` (jupyter, matplotlib, pandas) and `dev` (pytest). A filter model would later add scikit-learn.

**Why.** `ollama` already requires pydantic. pydantic replaces the previous 332-line hand-written config loader with declarations, and it also defines the output schemas (D6). The standard library covers downloads (`urllib`) and TOML (`tomllib`).

### D13. A small test file instead of a test suite

**Decision.** The large test suite of the previous system is gone. `tests/test_metrics.py` keeps seven hand-made checks of the metrics and the selection: flat metrics, ancestor closures, hierarchical partial credit, ranking metrics, the threshold and cap, tie order, and that a selection rule is required.

**Why.** The metrics produce the thesis numbers, and an error there changes them silently. Two fixes in the previous code review were metric bugs: codes split or truncated in hierarchical metrics, and failed images scored as zero. Everything else fails loudly when it breaks.

## 5. Where each research goal lands

| Goal | Lands in | What it takes |
|---|---|---|
| **Core goals** | | |
| Top-k → similarity threshold, or top-k with a cutoff | `selection.py`, `caris sweep` | no new tags |
| Hierarchical context in code texts | `codes.document_text`, a second index | one `caris index` |
| Kaggle or another free environment | `notebooks/kaggle_tag.ipynb` | only the tag stage moves |
| More VLMs | `[tagging].model` | one tag run per VLM and split |
| More embedding models | `[embedding]`: model and prefixes | one index per model |
| Enforced structured output | `schemas.py`, `tagging.py` | built in from the start (D6) |
| Filtering model | between `retrieval.py` and `selection.py` | tagged `train` images (GPU) |
| **Additional goals** | | |
| Dataset analysis, stratification by motif | breakdowns in `evaluate.py`, `analysis.ipynb` | per-image results already carry the true codes |
| Pre-iconographical / iconographical labels, salience, relations | a new schema version; salience becomes a query weight | one tag run per schema |
| Hierarchy-aware metrics | `metrics.py` | new functions, e.g. LCA-based measures (Kosmopoulos et al., 2015) |

The two code-text variants, for code `14A1`:

- `standalone`: `astrologer observing the heavens`
- `hierarchical`: `Religion and Magic → astrology → astrologer at work → astrologer observing the heavens`

The path (`1` → `14` → `14A` → `14A1`) comes from the `iconclass` tree, and each step's text from the English Iconclass texts. The tree is older than some of the texts: 93 codes are missing from it, and they take the path of their longest prefix that the tree has.

## 6. Producing the semester artifacts

1. **Precision-recall trade-off across thresholds.**
   - `caris sweep … --split val` writes one table with the top-k curve, the threshold curve and the capped threshold curves, and `analysis.ipynb` plots them together.
   - Choose τ on `val`, and show the curves on `test`.
2. **Standard vs hierarchy-aware embeddings.** `experiments/gemma4-26b__mxbai__standalone.toml` and `…__hierarchical.toml` differ only in `[embedding].text` and share one tag run. Compare the curves, the threshold-free ranking metrics and the per-branch breakdowns side by side.
3. **Model comparison table.** One experiment per VLM, with the same prompt, schema, embedding and selection rule; τ is chosen per model on `val`. Report P/R/F1, hP/hR/hF1 and hit rate with bootstrap intervals, plus the median seconds per image. This is the first table in `analysis.ipynb`.
4. **Filter impact.** The same candidates with and without the filter, at operating points chosen on `val`. Report the recall lost and the precision gained, next to the frequency baseline in §7.

## 7. Risks and early checks

- **Kaggle speed.** Time 20 images before a full run (`--limit 20`). Models that don't fit in the local 8 GB, such as Gemma 4 26B and 31B, are the ones most likely to gain. Small models such as `qwen3.5:0.8b` fit locally.
- **Run-to-run variation.** Tag 100 `test` images a second time with the same experiment, on another day or machine (about 40 extra GPU minutes). That shows how much tags vary when nothing changes. Differences between VLMs smaller than that variation aren't meaningful.
- **The filter's training data costs GPU time.** The filter needs VLM tags on `train` images: 2,000 images take about 13 h at 23 s each. Start from a free baseline.
  - About 70% of the index's codes never occur in the training labels.
  - On the previous system, dropping those codes from Gemma's predictions cut 1,222 codes per image to 725, with recall going from 43.5% to only 43.3%.
  - Keeping only codes seen at least 20 times gave 221 codes per image, 39.4% recall and 5× the precision.

  A learned filter has to beat that.
- **Hierarchical texts may blur siblings.** A long shared path makes sibling codes look alike. That can help hierarchical metrics and hurt exact precision. A "parent + code" text variant to compare costs one more index.
- **Enforced structure may change content (D6).** Run the prompt-only ablation once, with the main VLM.

## 8. Decided while building

| Question | Decision |
|---|---|
| The split | Seed 2026, 1,000 test and 400 val images, random rather than stratified, with per-branch reporting (D8) |
| Tests | One small file of metric and selection checks (D13) |
| Repository | A new repository, CARIS v2; the previous one is untouched (D10) |
| Where prompts live | Inside the package, so every installed copy has them |
| Where data lives | `./data` by default, or wherever `CARIS_DATA` points, as on Kaggle |
| Pooled metrics | Reported next to per-image means, because strict thresholds leave some images without codes |

## 9. Next steps

1. Tag `val` and `test` with `gemma4:26b`, on this machine or on Kaggle.
2. Sweep both code-text variants on `val`, choose τ for each, and add it to the experiment files.
3. Evaluate both variants on `test`: artifacts 1 and 2.
4. Add one experiment file per VLM to compare. `ornith-1.5:9b`, `qwen3.5:0.8b` and `gemma4:31b` are already pulled locally. Then tag, sweep and evaluate each one: artifact 3.
5. Try other embedding models, each with the query and document prefixes from its model card.
6. Tag a sample of `train` images and train the filter model against the frequency baseline: artifact 4.

## References

- Ollama documentation, [Structured outputs](https://docs.ollama.com/capabilities/structured-outputs).
- Kaggle GPU quota, from secondary sources (your Kaggle account settings show the actual figure): [Free cloud GPUs in 2026](https://gpuperhour.com/blog/free-cloud-gpus-and-credits), [Colab alternatives compared, 2026](https://www.spheron.network/blog/google-colab-alternatives-8-gpu-clouds-compared-2026/).
- S. Kiritchenko, S. Matwin, R. Nock, A. F. Famili (2006). *Learning and evaluation in the presence of class hierarchies: application to text categorization.* Canadian Conference on AI.
- A. Kosmopoulos, I. Partalas, E. Gaussier, G. Paliouras, I. Androutsopoulos (2015). *Evaluation measures for hierarchical classification: a unified view and novel approaches.* Data Mining and Knowledge Discovery.
- Z. R. Tam et al. (2024). *Let Me Speak Freely? A study on the impact of format restrictions on performance of large language models.*
- K. Sechidis, G. Tsoumakas, I. Vlahavas (2011). *On the stratification of multi-label data.* ECML PKDD.
