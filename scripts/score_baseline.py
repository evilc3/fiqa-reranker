"""Score validation/test candidates with the pinned pretrained model; no training."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path

import pandas as pd
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fiqa_reranker.ranking import load_qrels, load_split, ranking_metrics, score_dataset, seed_everything, select_device, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/step3.yaml"))
    parser.add_argument("--split", choices=("validation", "test"), help="Score only this split; never starts training")
    parser.add_argument("--output-dir", type=Path, help="New directory for scores, metrics, and resolved config")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.split:
        cfg["data"]["splits"] = [args.split]
    if args.output_dir:
        out_dir = args.output_dir
    elif args.split:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_utc")
        out_dir = Path(cfg["baseline"]["output_dir"]) / f"pretrained_{args.split}_{timestamp}"
    else:
        out_dir = Path(cfg["baseline"]["output_dir"])
    if (args.split or args.output_dir) and out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out_dir}; choose a new --output-dir")
    cfg["baseline"]["output_dir"] = str(out_dir)
    seed_everything(int(cfg["seed"]))
    model_cfg, data_cfg, base_cfg = cfg["model"], cfg["data"], cfg["baseline"]
    device, gpu_name = select_device(str(base_cfg.get("device", "cuda")))
    model_path = Path(model_cfg["local_dir"])
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Pinned local checkpoint missing: {model_path}; run Step 2.2 first")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    model.to(device).eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"config": cfg, "model_path": str(model_path.resolve()), "device": str(device), "gpu_name": gpu_name, "splits": {}}
    print(f"Using GPU: {gpu_name} ({device})" if gpu_name else f"CUDA unavailable; using CPU ({device})", flush=True)

    for split in data_cfg["splits"]:
        print(f"=== PRETRAINED BASELINE: SCORE {split.upper()} CANDIDATES ===", flush=True)
        dataset = load_split(Path(data_cfg["tokenized_dir"]), split)
        predictions = score_dataset(model, dataset, tokenizer, batch_size=int(base_cfg["batch_size"]), device=device, desc=f"{split}: scoring")
        predictions.to_csv(out_dir / f"{split}_scores.tsv", sep="\t", index=False)
        qrels_cfg = data_cfg["qrels"]
        qrels = load_qrels(qrels_cfg["name"], qrels_cfg["revision"], qrels_cfg["split_names"][split])
        metrics = ranking_metrics(predictions, qrels, k=10)
        results["splits"][split] = {**metrics, "rows_scored": len(predictions), "score_file": f"{split}_scores.tsv"}
        print(f"{split}: nDCG@10={metrics['ndcg@10']:.6f}, MRR@10={metrics['mrr@10']:.6f}; {len(predictions):,} candidates", flush=True)

    results["package_versions"] = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "datasets", "pandas")}
    write_json(out_dir / "metrics.json", results)
    (out_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"Scores and metrics saved to: {out_dir.resolve()}", flush=True)
    print(f"Metrics JSON: {(out_dir / 'metrics.json').resolve()}", flush=True)


if __name__ == "__main__":
    main()
