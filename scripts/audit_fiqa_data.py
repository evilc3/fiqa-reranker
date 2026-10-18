"""Audit processed and tokenized FiQA training/evaluation pairs; does not train."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from fiqa_reranker.data_quality import audit_fiqa_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/step3.yaml"))
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--tokenized-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/data_quality"))
    parser.add_argument("--examples-per-category", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_cfg, model_cfg = cfg["data"], cfg["model"]
    qrels_cfg = data_cfg["qrels"]
    split_names = qrels_cfg.get("split_names", {})
    qrels_splits = {
        split: split_names.get(split, split)
        for split in ("train", "validation", "test")
    }
    report = audit_fiqa_data(
        tokenized_dir=args.tokenized_dir or Path(data_cfg["tokenized_dir"]),
        processed_dir=args.processed_dir or Path(data_cfg.get("processed_dir", data_cfg.get("qrels_dir", "data/processed/fiqa_fast"))),
        model_dir=Path(model_cfg["local_dir"]),
        qrels_name=qrels_cfg["name"],
        qrels_revision=qrels_cfg["revision"],
        qrels_splits=qrels_splits,
        output_dir=args.output_dir,
        seed=int(cfg["seed"]),
        examples_per_category=args.examples_per_category,
        batch_size=args.batch_size,
    )
    print("=== FIQA DATA QUALITY AUDIT COMPLETE ===", flush=True)
    for split, stats in report["split_reports"].items():
        print(
            f"{split}: rows={stats['tokenized_rows']:,}; "
            f"alignment issues={stats['alignment_issue_count']}; "
            f"label issues={stats['label_issue_count']}; "
            f"truncated={stats['truncated_rows']:,} ({stats['truncation_fraction']:.2%})",
            flush=True,
        )
        for row in stats["truncation_by_label_and_provenance"]:
            print(
                f"  {row['label_class']} / {row['negative_provenance']}: "
                f"{row['truncated_rows']:,}/{row['rows']:,} truncated ({row['fraction']:.2%})",
                flush=True,
            )
    print(f"Total alignment issues: {report['alignment_issue_count']}", flush=True)
    print(f"Total label/qrels issues: {report['label_issue_count']}", flush=True)
    print(f"Duplicate query/document pair rows: {report['duplicate_pair_row_count']}", flush=True)
    print(f"Audit summary: {(args.output_dir / 'audit_summary.json').resolve()}", flush=True)
    for filename in report["outputs"].values():
        print(f"Output: {(args.output_dir / filename).resolve()}", flush=True)


if __name__ == "__main__":
    main()
