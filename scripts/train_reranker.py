"""Fine-tune and evaluate the FiQA cross-encoder. This script runs only when invoked."""

from __future__ import annotations

import argparse
import importlib.metadata
import math
import re
from datetime import datetime
from pathlib import Path
from secrets import choice as random_choice

import pandas as pd
import torch
from torch import nn
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from fiqa_reranker.ranking import (
    load_qrels, load_split, make_collator, ranking_metrics,
    score_dataset, seed_everything, select_device, write_json,
)

_ANIMALS = (
    "badger", "beaver", "bison", "falcon", "fox", "gecko", "heron",
    "ibis", "jaguar", "lemur", "lynx", "otter", "panda", "quokka",
    "raven", "tiger", "wombat", "yak",
)


def make_run_name(requested_name: str | None) -> str:
    """Return a filesystem-safe explicit name or a timestamped animal name."""
    if requested_name:
        name = re.sub(r"[^A-Za-z0-9._-]+", "-", requested_name).strip("-._")
        if not name:
            raise ValueError("--training-run-name must contain at least one letter or digit")
        return name
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{random_choice(_ANIMALS)}"


def make_optimizer(model: torch.nn.Module, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "layernorm.weight")
    decay_params, no_decay_params = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay_params if any(key in name for key in no_decay) else decay_params).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": weight_decay}, {"params": no_decay_params, "weight_decay": 0.0}],
        lr=learning_rate,
    )


def evaluate(model, tokenizer, dataset, qrels, batch_size: int, device: torch.device, desc: str):
    predictions = score_dataset(model, dataset, tokenizer, batch_size=batch_size, device=device, desc=desc)
    metrics = ranking_metrics(predictions, qrels, k=10)
    return predictions, metrics


def run_overfit(model_path: Path, tokenizer, data_cfg: dict, train_cfg: dict, seed: int, device: torch.device, output_dir: Path) -> None:
    """Diagnostic on one positive batch; mutated weights are discarded at the end."""
    seed_everything(seed)
    original_model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).to(device)
    original_state = {key: value.detach().cpu().clone() for key, value in original_model.state_dict().items()}
    del original_model
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).to(device)
    train = load_split(Path(data_cfg["tokenized_dir"]), "train")
    labeled = train.filter(lambda row: row["label"] is not None, desc="Overfit diagnostic: filter labels")
    one_batch = next(iter(DataLoader(labeled, batch_size=int(train_cfg["batch_size"]), shuffle=False, collate_fn=make_collator(tokenizer))))
    inputs = {key: value.to(device) for key, value in one_batch["inputs"].items()}
    labels = one_batch["labels"].to(device)
    optimizer = make_optimizer(model, float(train_cfg["overfit"]["learning_rate"]), 0.0)
    criterion = nn.BCEWithLogitsLoss()
    steps = int(train_cfg["overfit"]["steps"])
    rows = []
    for step in tqdm(range(1, steps + 1), desc="One-batch overfit", unit="step", dynamic_ncols=True):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(**inputs).logits.reshape(-1).float()
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["gradient_clip_norm"]))
        optimizer.step()
        rows.append({"step": step, "bce_loss": float(loss.detach().cpu())})
    model.load_state_dict(original_state)
    check = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).to(device)
    if any(not torch.equal(a.detach().cpu(), b.detach().cpu()) for a, b in zip(model.state_dict().values(), check.state_dict().values(), strict=True)):
        raise RuntimeError("One-batch diagnostic failed to restore the original pretrained weights")
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "overfit_loss.tsv", sep="\t", index=False)
    del check, model
    print(f"Overfit diagnostic complete; original checkpoint remains unchanged. Loss trace: {(output_dir / 'overfit_loss.tsv').resolve()}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/step3.yaml"))
    parser.add_argument("--overfit-one-batch", action="store_true", help="Run the isolated one-batch diagnostic only; no training run")
    parser.add_argument("--skip-baseline-validation", action="store_true", help="Skip scoring the untouched pretrained model on validation before training")
    parser.add_argument("--training-run-name", type=str, help="Name for this run's output subdirectory; defaults to timestamp_random-animal")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_everything(int(cfg["seed"]))
    model_cfg, data_cfg, train_cfg = cfg["model"], cfg["data"], cfg["training"]
    model_path = Path(model_cfg["local_dir"])
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Pinned local checkpoint missing: {model_path}; run Step 2.2 first")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    device, gpu_name = select_device(str(cfg.get("device", "auto")))
    run_name = make_run_name(args.training_run_name)
    output_root = Path(train_cfg["output_dir"])
    output_dir = output_root / run_name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Training run output already exists and is not empty: {output_dir}; choose another --training-run-name")
    print(f"Training run name: {run_name}", flush=True)
    print(f"Training run output: {output_dir.resolve()}", flush=True)
    if args.overfit_one_batch:
        run_overfit(model_path, tokenizer, data_cfg, train_cfg, int(cfg["seed"]), device, output_dir / "overfit_diagnostic")
        return

    train = load_split(Path(data_cfg["tokenized_dir"]), "train")
    train = train.filter(lambda row: row["label"] is not None, desc="Training: exclude null labels")
    validation = load_split(Path(data_cfg["tokenized_dir"]), "validation")
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).to(device)
    batch_size = int(train_cfg["batch_size"])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(int(cfg["seed"])), collate_fn=make_collator(tokenizer), num_workers=int(train_cfg.get("num_workers", 0)))
    optimizer = make_optimizer(model, float(train_cfg["learning_rate"]), float(train_cfg["weight_decay"]))
    updates_per_epoch = len(train_loader)
    total_steps = updates_per_epoch * int(train_cfg["max_epochs"])
    warmup_steps = int(total_steps * float(train_cfg["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    amp_enabled = bool(train_cfg.get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    criterion = nn.BCEWithLogitsLoss()
    qrels_cfg = data_cfg["qrels"]
    validation_qrels = load_qrels(qrels_cfg["name"], qrels_cfg["revision"], qrels_cfg["split_names"]["validation"])
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_cfg = {**cfg, "training": {**train_cfg, "run_name": run_name, "output_dir": str(output_dir.resolve())}}
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved_cfg, sort_keys=False), encoding="utf-8")

    if args.skip_baseline_validation:
        print("Skipping pre-training baseline validation as requested.", flush=True)
    else:
        # Score untouched pretrained weights on the same validation candidates
        # and qrels used after every epoch; the test split remains untouched.
        baseline_dir = output_dir / "pretraining_baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        print("=== PRE-TRAINING BASELINE: untouched pretrained model on validation ===", flush=True)
        baseline_predictions, baseline_metrics = evaluate(
            model, tokenizer, validation, validation_qrels, batch_size, device,
            "Pre-training baseline: validation",
        )
        baseline_predictions.to_csv(baseline_dir / "validation_scores.tsv", sep="\t", index=False)
        write_json(baseline_dir / "validation_metrics.json", {
            **baseline_metrics,
            "checkpoint": str(model_path.resolve()),
            "split": "validation",
            "note": "Measured before any training updates; uses original qrels and the same candidates as epoch validation.",
        })
        print(
            f"Pre-training baseline: validation nDCG@10={baseline_metrics['ndcg@10']:.6f}, "
            f"MRR@10={baseline_metrics['mrr@10']:.6f}; metrics: "
            f"{(baseline_dir / 'validation_metrics.json').resolve()}",
            flush=True,
        )

    history: list[dict] = []
    best_ndcg, best_epoch, stale = -float("inf"), 0, 0

    device_summary = f"GPU {gpu_name} ({device})" if gpu_name else f"CPU ({device}); CUDA unavailable"
    print(f"Training {len(train):,} labeled pairs on {device_summary}; batch={batch_size}; max epochs={train_cfg['max_epochs']}", flush=True)
    for epoch in range(1, int(train_cfg["max_epochs"]) + 1):
        model.train()
        losses: list[float] = []
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{train_cfg['max_epochs']}: train", unit="batch", dynamic_ncols=True)
        for batch in progress:
            inputs = {key: value.to(device) for key, value in batch["inputs"].items()}
            labels = batch["labels"].to(device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(**inputs).logits.reshape(-1).float()
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            progress.set_postfix(bce=f"{losses[-1]:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        print(f"Epoch {epoch}: scoring validation candidates", flush=True)
        validation_predictions, metrics = evaluate(model, tokenizer, validation, validation_qrels, batch_size, device, f"Epoch {epoch}: validation")
        validation_predictions.to_csv(output_dir / f"validation_epoch_{epoch:02d}_scores.tsv", sep="\t", index=False)
        row = {"epoch": epoch, "train_bce_loss": sum(losses) / max(1, len(losses)), "validation_ndcg@10": metrics["ndcg@10"], "validation_mrr@10": metrics["mrr@10"], "validation_queries": metrics["queries"], "learning_rate": scheduler.get_last_lr()[0]}
        history.append(row)
        checkpoint = output_dir / f"checkpoint_epoch_{epoch:02d}"
        if bool(train_cfg.get("save_every_epoch", True)):
            model.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
        if float(metrics["ndcg@10"]) > best_ndcg:
            best_ndcg, best_epoch, stale = float(metrics["ndcg@10"]), epoch, 0
            model.save_pretrained(output_dir / "best_model", safe_serialization=True)
            tokenizer.save_pretrained(output_dir / "best_model")
        else:
            stale += 1
        write_json(output_dir / "training_history.json", {"epochs": history, "best_epoch": best_epoch, "best_validation_ndcg@10": best_ndcg})
        print(f"Epoch {epoch}: train BCE={row['train_bce_loss']:.6f}; validation nDCG@10={metrics['ndcg@10']:.6f}, MRR@10={metrics['mrr@10']:.6f}; stale={stale}/{train_cfg['patience']}", flush=True)
        if stale >= int(train_cfg["patience"]):
            print("Early stopping: validation nDCG@10 did not improve.", flush=True)
            break

    print(f"Restoring best checkpoint from epoch {best_epoch}", flush=True)
    best_model = AutoModelForSequenceClassification.from_pretrained(output_dir / "best_model", local_files_only=True).to(device)
    test = load_split(Path(data_cfg["tokenized_dir"]), "test")
    test_predictions = score_dataset(best_model, test, tokenizer, batch_size=batch_size, device=device, desc="Best checkpoint: test scoring")
    test_predictions.to_csv(output_dir / "test_scores.tsv", sep="\t", index=False)
    test_qrels = load_qrels(qrels_cfg["name"], qrels_cfg["revision"], qrels_cfg["split_names"]["test"])
    test_metrics = ranking_metrics(test_predictions, test_qrels, k=10)
    write_json(output_dir / "test_metrics.json", {**test_metrics, "best_epoch": best_epoch, "checkpoint": "best_model", "note": "One final test evaluation after validation-based model selection; do not tune using these results."})
    metadata = {"run_name": run_name, "baseline_validation_skipped": args.skip_baseline_validation, "model": model_cfg, "seed": cfg["seed"], "device": str(device), "gpu_name": gpu_name, "best_epoch": best_epoch, "test_metrics": test_metrics, "package_versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "datasets", "pandas")}}
    write_json(output_dir / "run_metadata.json", metadata)
    print(f"Final test: nDCG@10={test_metrics['ndcg@10']:.6f}, MRR@10={test_metrics['mrr@10']:.6f}", flush=True)
    print(f"Training artifacts saved to: {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
