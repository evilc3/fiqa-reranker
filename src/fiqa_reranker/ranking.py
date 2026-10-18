"""Ranking metrics and data helpers shared by FiQA Step 3 scripts."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(configured_device: str) -> tuple[torch.device, str | None]:
    """Use CUDA when requested and available; otherwise make CPU fallback explicit."""
    if configured_device not in {"auto", "cuda", "cpu"}:
        raise ValueError("device must be one of: auto, cuda, cpu")
    if configured_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable. Check the NVIDIA driver and CUDA-enabled PyTorch install.")
    if configured_device != "cpu" and torch.cuda.is_available():
        device = torch.device("cuda:0")
        return device, torch.cuda.get_device_name(device)
    return torch.device("cpu"), None


def load_qrels(name: str, revision: str, split_name: str) -> dict[str, dict[str, float]]:
    """Load original official qrels directly from their pinned HF snapshot."""
    frame = load_dataset(name, split=split_name, revision=revision).to_pandas()
    expected = {"query-id", "corpus-id", "score"}
    if not expected.issubset(frame.columns):
        raise ValueError(f"Official qrels missing {sorted(expected - set(frame.columns))}")
    frame = frame.rename(columns={"query-id": "query_id", "corpus-id": "doc_id"})
    frame["query_id"] = frame.query_id.astype(str)
    frame["doc_id"] = frame.doc_id.astype(str)
    frame["score"] = pd.to_numeric(frame.score, errors="raise")
    if frame.duplicated(["query_id", "doc_id"]).any():
        conflicts = frame.groupby(["query_id", "doc_id"]).score.nunique()
        if (conflicts > 1).any():
            raise ValueError("Conflicting relevance judgments in official qrels")
        frame = frame.drop_duplicates(["query_id", "doc_id"])
    qrels: dict[str, dict[str, float]] = {}
    for row in frame.itertuples(index=False):
        qrels.setdefault(row.query_id, {})[row.doc_id] = float(row.score)
    return qrels


def ranking_metrics(
    predictions: pd.DataFrame,
    qrels: dict[str, dict[str, float]],
    *,
    k: int = 10,
) -> dict[str, float | int]:
    """Compute query-averaged nDCG@k and MRR@k against original qrels.

    Candidate documents without qrels remain rankable but have unknown relevance;
    they are omitted from metric gains, rather than interpreted as negatives.
    The ideal DCG uses all positive qrels for each query, including unretrieved ones.
    """
    ndcgs: list[float] = []
    reciprocal_ranks: list[float] = []
    grouped = {str(qid): group.sort_values("score", ascending=False, kind="stable") for qid, group in predictions.groupby("query_id", sort=False)}
    for qid, judgments in qrels.items():
        if not any(score > 0 for score in judgments.values()):
            continue
        group = grouped.get(str(qid))
        ranked_ids = [] if group is None else group.doc_id.astype(str).tolist()[:k]
        dcg = sum((2**max(0.0, judgments[doc_id]) - 1) / math.log2(rank + 2) for rank, doc_id in enumerate(ranked_ids) if doc_id in judgments)
        ideal = sorted((max(0.0, value) for value in judgments.values()), reverse=True)[:k]
        idcg = sum((2**gain - 1) / math.log2(rank + 2) for rank, gain in enumerate(ideal))
        ndcgs.append(dcg / idcg if idcg else 0.0)
        reciprocal_ranks.append(next((1.0 / (rank + 1) for rank, doc_id in enumerate(ranked_ids) if judgments.get(doc_id, 0) > 0), 0.0))
    return {"ndcg@10": float(np.mean(ndcgs)) if ndcgs else 0.0, "mrr@10": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0, "queries": len(ndcgs), "k": k}


def make_collator(tokenizer):
    def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        model_inputs = [{key: row[key] for key in ("input_ids", "attention_mask", "token_type_ids") if key in row and row[key] is not None} for row in rows]
        padded = tokenizer.pad(model_inputs, padding=True, return_tensors="pt")
        return {
            "inputs": padded,
            # Null labels are expected for unjudged validation/test candidates.
            # They are represented as NaN only in this transient batch tensor;
            # baseline scoring ignores labels, and training filters nulls first.
            "labels": torch.tensor(
                [float(row["label"]) if row.get("label") is not None else float("nan") for row in rows],
                dtype=torch.float32,
            ),
            "query_ids": [str(row["query_id"]) for row in rows],
            "doc_ids": [str(row["doc_id"]) for row in rows],
            "is_judged": torch.tensor([int(row["is_judged"]) for row in rows]),
            "relevance_scores": [row.get("relevance_score") for row in rows],
        }
    return collate


def load_split(root: Path, split: str):
    return load_from_disk(str(root / split))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def score_dataset(model, dataset, tokenizer, *, batch_size: int, device: torch.device, desc: str) -> pd.DataFrame:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=make_collator(tokenizer))
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True):
            inputs = {key: value.to(device) for key, value in batch["inputs"].items()}
            logits = model(**inputs).logits.reshape(-1).float().cpu().tolist()
            records.extend({"query_id": qid, "doc_id": did, "score": float(score)} for qid, did, score in zip(batch["query_ids"], batch["doc_ids"], logits, strict=True))
    return pd.DataFrame.from_records(records, columns=["query_id", "doc_id", "score"])


def score_labeled_dataset(model, dataset, tokenizer, *, batch_size: int, device: torch.device, desc: str) -> pd.DataFrame:
    """Score judged candidates only for validation metrics; unknowns aren't zeros."""
    selected = dataset.filter(lambda row: int(row["is_judged"]) == 1 and row["relevance_score"] is not None, desc=f"{desc}: retain judged")
    predictions = score_dataset(model, selected, tokenizer, batch_size=batch_size, device=device, desc=desc)
    predictions["relevance_score"] = selected["relevance_score"]
    return predictions
