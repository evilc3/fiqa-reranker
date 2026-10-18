"""Download the pinned cross-encoder and tokenize FiQA pair TSVs; does not train."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from fiqa_reranker.tokenization import load_pinned_model, tokenize_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-length", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_cfg, token_cfg = config["model"], config["tokenization"]
    model_dir = Path(model_cfg["local_dir"])
    input_dir = args.input_dir or Path(token_cfg["input_dir"])
    output_dir = args.output_dir or Path(token_cfg["output_dir"])
    max_length = args.max_length or int(model_cfg["max_length"])

    print("=== STEP 2.2: MODEL DOWNLOAD AND OFFLINE RELOAD ===")
    tokenizer = load_pinned_model(model_cfg["name"], model_cfg["revision"], model_dir)
    print("=== STEP 2.2: TOKENIZE QUERY–PASSAGE PAIRS ===")
    manifest = tokenize_splits(
        input_dir=input_dir,
        output_dir=output_dir,
        tokenizer=tokenizer,
        max_length=max_length,
        seed=int(config["seed"]),
        model_name=model_cfg["name"],
        revision=model_cfg["revision"],
        local_model_dir=model_dir,
        batch_size=int(token_cfg.get("batch_size", 512)),
    )
    print("=== TOKENIZATION COMPLETE (NO TRAINING WAS RUN) ===")
    print(f"Model snapshot: {model_dir.resolve()}")
    print(f"Tokenized datasets: {output_dir.resolve()}")
    print(f"Manifest: {(output_dir.resolve() / 'manifest.json')}")
    print(f"Overlong query exclusions: {(output_dir.resolve() / 'exclusions.jsonl')}")
    for split, stats in manifest["splits"].items():
        print(
            f"{split}: {stats['saved_rows']:,} rows; passage truncation "
            f"{stats['passages_truncated']:,} ({stats['passage_truncation_fraction']:.2%}); "
            f"overlong queries rejected {stats['rejected_overlong_queries']:,}"
        )
    print("Output SHA-256 entries:", len(manifest["output_sha256"]))


if __name__ == "__main__":
    main()
