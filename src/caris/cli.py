"""The caris command: split, index, tag, sweep, evaluate and predict."""

import argparse
import sys
from pathlib import Path

import ollama
import pydantic

from caris import data, evaluate, index, tagging
from caris.config import load_experiment
from caris.predict import predict

SPLITS = ("val", "test")


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(errors="replace")  # Iconclass texts hold characters some consoles lack
    args = _parser().parse_args(argv)
    try:
        args.run(args)
    except (ConnectionError, ollama.ResponseError, pydantic.ValidationError,
            FileNotFoundError, LookupError, RuntimeError) as error:
        print(f"caris: {error}", file=sys.stderr)
        return 1
    return 0


def run_split(args):
    sizes = data.split_dataset(args.zip)
    print(", ".join(f"{name}: {size} images" for name, size in sizes.items()))


def run_index(args):
    path = index.build_index(load_experiment(args.experiment).embedding)
    print(f"Index written to {path}")


def run_tag(args):
    experiment = load_experiment(args.experiment)
    path = tagging.tag_split(experiment.tagging, args.split or experiment.data.split, args.limit)
    print(f"Tag run: {path}")


def run_sweep(args):
    experiment = load_experiment(args.experiment)
    path = evaluate.sweep(experiment, args.split or experiment.data.split, args.thresholds, args.top_k)
    print(f"Sweep written to {path}")


def run_evaluate(args):
    experiment = load_experiment(args.experiment)
    path, results = evaluate.evaluate(experiment, args.split or experiment.data.split)
    images, summary = results["images"], results["summary"]
    print(f"{images['evaluated']} images evaluated, {len(images['failed'])} failed, "
          f"{len(images['untagged'])} not tagged yet.")
    for name in ("hit", "precision", "recall", "f1", "h_f1"):
        if name in summary:
            low, high = summary[name]["ci95"]
            print(f"  {name:<10} {summary[name]['mean']:.4f}  (95% interval {low:.4f} to {high:.4f})")
    print(f"Results written to {path}")


def run_predict(args):
    result = predict(args.image, load_experiment(args.experiment))
    print("Tags:", ", ".join(result.tags))
    print("\nCodes:")
    for code, score, text in result.codes:
        print(f"  {score:.3f}  {code:<16} {text}")
    print("\nSimilar artworks in the train split:")
    for recommender, artwork in result.recommendations.items():
        print(f"  {recommender:<8} {artwork or '(none)'}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="caris", description="Iconclass codes and similar artworks from a local VLM.")
    commands = parser.add_subparsers(required=True, metavar="command")

    split = commands.add_parser("split", help="split the dataset into train, val and test (once)")
    split.add_argument("--zip", type=Path, default=data.DATA / "iconclass_dataset.zip",
                       help="the Iconclass test set archive (default: data/iconclass_dataset.zip)")
    split.set_defaults(run=run_split)

    def with_experiment(name, run, description, takes_split=True):
        command = commands.add_parser(name, help=description)
        command.add_argument("experiment", type=Path, help="an experiment file, such as experiments/x.toml")
        if takes_split:
            command.add_argument("--split", choices=SPLITS, help="default: the experiment's [data] split")
        command.set_defaults(run=run)
        return command

    with_experiment("index", run_index, "embed every code text for the experiment's embedding setup", takes_split=False)
    tag = with_experiment("tag", run_tag, "tag a split with the experiment's VLM; resumes where it stopped")
    tag.add_argument("--limit", type=int, help="tag only the split's first N images")
    sweep = with_experiment("sweep", run_sweep, "precision and recall for every pair of threshold and top-k cap")
    sweep.add_argument("--thresholds", type=_numbers(float), default="none,0.55:0.95:0.01",
                       help="comma-separated; START:STOP:STEP is a range, none means no threshold "
                            "(default: none,0.55:0.95:0.01)")
    sweep.add_argument("--top-k", type=_numbers(int), default="none,1,2,5,10,20,50",
                       help="comma-separated caps; none means no cap (default: none,1,2,5,10,20,50)")
    with_experiment("evaluate", run_evaluate, "score the experiment's selection rule and write the results")

    predict_command = commands.add_parser("predict", help="codes and similar artworks for one image")
    predict_command.add_argument("image", type=Path)
    predict_command.add_argument("--experiment", type=Path, required=True)
    predict_command.set_defaults(run=run_predict)
    return parser


def _numbers(convert):
    """An argparse type for a comma-separated list, where START:STOP:STEP is a range and none is None."""

    def parse(text: str) -> list:
        values = []
        for item in text.split(","):
            if item.strip() == "none":
                values.append(None)
            elif ":" in item:
                start, stop, step = map(float, item.split(":"))
                values += [round(start + i * step, 4) for i in range(round((stop - start) / step) + 1)]
            else:
                values.append(convert(item))
        return values

    return parse
