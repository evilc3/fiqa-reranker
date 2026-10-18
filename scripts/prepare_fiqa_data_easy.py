"""Create easy-only training pairs and preserve existing evaluation TSVs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from fiqa_reranker.data_pipeline import load_huggingface_bundle
from fiqa_reranker.easy_pipeline import export_easy_pipeline, verify_evaluation_source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data_easy.yaml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preview-rows", type=int, default=3)
    args = parser.parse_args()
    if args.preview_rows < 0:
        parser.error("--preview-rows cannot be negative")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.output_dir:
        config["output_dir"] = str(args.output_dir)
    source_dir, output_dir = Path(config["evaluation_source_dir"]), Path(config["output_dir"])
    if source_dir.resolve() == output_dir.resolve() or output_dir.resolve() in source_dir.resolve().parents or source_dir.resolve() in output_dir.resolve().parents:
        parser.error("Output must be separate from the evaluation source directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {output_dir}; choose a new --output-dir")
    verify_evaluation_source(source_dir, config)
    logging.info("Stage 1/5: load pinned corpus/queries/qrels and validate official splits")
    bundle, fingerprints = load_huggingface_bundle(config)
    manifest = export_easy_pipeline(
        bundle, config=config, fingerprints=fingerprints,
        source_dir=source_dir, output_dir=output_dir,
    )
    print("=== EASY-ONLY PREPROCESSING COMPLETE ===", flush=True)
    print(f"Output directory: {output_dir.resolve()}", flush=True)
    print(f"Rows by split: {manifest['counts']['examples']}", flush=True)
    print(
        f"Train: {manifest['counts']['train_positives']:,} positives; "
        f"{manifest['counts']['train_negatives']:,} negatives; "
        f"0 duplicate query/document pairs; ratio 1:{config['sampling']['negatives_per_positive']}",
        flush=True,
    )
    for filename in [*manifest["artifacts"]["files"].values(), "manifest.json"]:
        print(f"Saved: {(output_dir / filename).resolve()}", flush=True)
    print("Validation/test are byte-identical to the original evaluation files.", flush=True)
    if args.preview_rows:
        preview = pd.read_csv(output_dir / "train.tsv", sep="\t", nrows=args.preview_rows)
        with pd.option_context("display.max_colwidth", 100, "display.width", 180):
            print("\n--- train.tsv preview ---", flush=True)
            print(preview.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
