"""Create reproducible FiQA training pairs and retrieved evaluation lists."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from fiqa_reranker.data_pipeline import (
    export_pipeline,
    load_huggingface_bundle,
)

LOGGER = logging.getLogger("fiqa_reranker.prepare_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--output-dir", type=Path, help="Override config output_dir")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--preview-rows", type=int, default=5, help="Print this many sample rows from each TSV (0 disables previews)")
    return parser.parse_args()


def main(*, fast: bool = False) -> None:
    args = parse_args()
    if args.preview_rows < 0:
        raise ValueError("--preview-rows must be zero or greater")
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = config["dataset"]
    retrieval = config["retrieval"]
    sampling = config["sampling"]
    if retrieval.get("method") != "bm25":
        raise ValueError("This pipeline currently supports retrieval.method: bm25")

    LOGGER.info("Stage 0/6: load pinned FiQA snapshots and validate dataset integrity")
    if fast:
        LOGGER.info("Fast mode: reuse query-level exclusions and hard candidates; rejection-sample easy negatives")
    bundle, fingerprints = load_huggingface_bundle(config)
    output_dir = args.output_dir or Path(config["output_dir"])
    if fast and args.output_dir is None:
        output_dir = output_dir.with_name(f"{output_dir.name}_fast")
    revisions = {
        "corpus_and_queries": f"{data['name']}@{data['revision']}",
        "qrels": f"{data['qrels']['name']}@{data['qrels']['revision']}",
    }
    manifest = export_pipeline(
        bundle,
        out_dir=output_dir,
        seed=int(config["seed"]),
        retrieval_depth=int(retrieval["training_depth"]),
        eval_top_k=int(retrieval["evaluation_top_k"]),
        bm25_k1=float(retrieval["k1"]),
        bm25_b=float(retrieval["b"]),
        negatives_per_positive=int(sampling["negatives_per_positive"]),
        hard_negative_fraction=float(sampling["hard_negative_fraction"]),
        dataset_revisions=revisions,
        fingerprints=fingerprints,
        config=config,
        fast_mode=fast,
    )
    print("\n=== PREPROCESSING COMPLETE ===")
    print(f"Output directory: {output_dir.resolve()}")
    print("Saved files:")
    for name in ("train", "validation", "test", "exclusions"):
        filename = manifest["artifacts"]["files"][name]
        row_count = manifest["counts"]["examples"].get(name)
        if name == "exclusions":
            row_count = len(bundle.exclusions)
        print(f"  {output_dir.joinpath(filename).resolve()} ({row_count:,} rows)")
    print(f"  {output_dir.joinpath('manifest.json').resolve()} (run configuration, counts, provenance, and checksums)")
    print(f"Rows by split: {manifest['counts']['examples']}")
    print(f"Exclusions by reason: {manifest['counts']['exclusions_by_reason']}")

    if args.preview_rows:
        preview_columns = [
            "split", "query_id", "doc_id", "label", "is_judged",
            "negative_provenance", "candidate_rank", "query_text", "doc_text",
        ]
        for name in ("train", "validation", "test", "exclusions"):
            path = output_dir / manifest["artifacts"]["files"][name]
            preview = pd.read_csv(path, sep="\t", nrows=args.preview_rows)
            print(f"\n--- {name}.tsv: first {len(preview)} row(s) ---")
            if name != "exclusions":
                preview = preview[[column for column in preview_columns if column in preview.columns]].copy()
                for column in ("query_text", "doc_text"):
                    if column in preview:
                        preview[column] = preview[column].fillna("").map(lambda value: str(value)[:120] + ("..." if len(str(value)) > 120 else ""))
            print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
