"""Pinned cross-encoder download and variable-length FiQA pair tokenization."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, Features, Sequence, Value
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LOGGER = logging.getLogger(__name__)

TOKEN_FEATURES = Features(
    {
        "pair_id": Value("string"),
        "split": Value("string"),
        "row_index": Value("int64"),
        "query_id": Value("string"),
        "doc_id": Value("string"),
        "input_ids": Sequence(Value("int32")),
        "attention_mask": Sequence(Value("int8")),
        "token_type_ids": Sequence(Value("int8")),
        "label": Value("float32"),
        "is_judged": Value("int8"),
        "relevance_score": Value("float32"),
        "negative_provenance": Value("string"),
        "candidate_source": Value("string"),
        "candidate_rank": Value("int32"),
        "retrieval_score": Value("float32"),
        "seed": Value("int64"),
    }
)


def _optional(row: dict[str, Any], key: str) -> Any:
    value = row.get(key)
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def load_pinned_model(model_name: str, revision: str, local_dir: Path):
    """Download model and tokenizer once; validate both reload fully offline."""
    local_dir = local_dir.resolve()
    if (local_dir / "tokenizer_config.json").exists() and (local_dir / "config.json").exists():
        LOGGER.info("Loading cached model and tokenizer from %s", local_dir)
        tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(local_dir, local_files_only=True)
    else:
        LOGGER.info("Downloading model and tokenizer %s at revision %s", model_name, revision)
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, revision=revision)
        local_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(local_dir)
        model.save_pretrained(local_dir, safe_serialization=True)

    LOGGER.info("Verifying offline tokenizer reload")
    offline = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
    offline_model = AutoModelForSequenceClassification.from_pretrained(local_dir, local_files_only=True)
    if offline.get_vocab() != tokenizer.get_vocab():
        raise RuntimeError("Offline tokenizer reload vocabulary does not match downloaded tokenizer")
    if (
        offline_model.config.model_type != model.config.model_type
        or offline_model.config.num_labels != model.config.num_labels
        or offline_model.config.vocab_size != model.config.vocab_size
    ):
        raise RuntimeError("Offline model reload architecture does not match downloaded model")
    del model, offline_model
    return offline


def tokenize_splits(
    *,
    input_dir: Path,
    output_dir: Path,
    tokenizer,
    max_length: int,
    seed: int,
    model_name: str,
    revision: str,
    local_model_dir: Path,
    batch_size: int = 512,
) -> dict[str, Any]:
    """Tokenize TSV pairs to split-specific Arrow datasets with no padding."""
    if max_length <= 0 or batch_size <= 0:
        raise ValueError("max_length and batch_size must be positive")
    if tokenizer.padding_side not in {"left", "right"}:
        raise ValueError(f"Unexpected padding side: {tokenizer.padding_side}")
    tokenizer_declared_max_length = tokenizer.model_max_length
    # We intentionally inspect full passage lengths to report truncation. Actual
    # pair encoding below always supplies max_length and only_second explicitly.
    tokenizer.model_max_length = max(int(tokenizer.model_max_length), 1_000_000)
    input_dir, output_dir = input_dir.resolve(), output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    special_count = tokenizer.num_special_tokens_to_add(pair=True)
    split_stats: dict[str, Any] = {}
    exclusions: list[dict[str, Any]] = []

    for split in ("train", "validation", "test"):
        source = input_dir / f"{split}.tsv"
        if not source.is_file():
            raise FileNotFoundError(f"Missing {split} TSV: {source}")
        frame = pd.read_csv(source, sep="\t", keep_default_na=True)
        required = {"query_id", "doc_id", "query_text", "doc_text", "label", "is_judged"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{source} missing required columns: {sorted(missing)}")
        if frame[["query_text", "doc_text"]].isna().any().any():
            raise ValueError(f"{source} contains missing query or document text")
        if split == "train" and frame["label"].isna().any():
            raise ValueError("Training labels must be present")
        observed_labels = set(frame["label"].dropna().astype(float).unique())
        if not observed_labels.issubset({0.0, 1.0}):
            raise ValueError(f"{split} labels must be binary where present; found {sorted(observed_labels)}")
        if split != "train" and frame.loc[frame["is_judged"].eq(0), "label"].notna().any():
            raise ValueError("Unjudged evaluation candidates unexpectedly have labels")

        query_lengths: dict[str, int] = {}
        for query in tqdm(frame["query_text"].drop_duplicates().tolist(), desc=f"{split}: check query lengths", unit="query", dynamic_ncols=True):
            query_ids = tokenizer(str(query), add_special_tokens=False, truncation=False)["input_ids"]
            query_lengths[str(query)] = len(query_ids)
        allowed_query_tokens = max_length - special_count
        valid_mask = pd.Series(
            [query_lengths[str(value)] <= allowed_query_tokens for value in frame["query_text"]],
            index=frame.index,
        )
        rejected = frame.loc[~valid_mask]
        for source_index, row in rejected.iterrows():
            exclusions.append({
                "split": split,
                "row_index": int(source_index),
                "query_id": str(row.query_id),
                "doc_id": str(row.doc_id),
                "reason": "overlong_query",
                "query_tokens": query_lengths[str(row.query_text)],
                "query_token_limit": allowed_query_tokens,
            })
        frame = frame.loc[valid_mask].copy()
        frame["_source_row"] = frame.index.astype("int64")
        frame = frame.reset_index(drop=True)
        records: list[dict[str, Any]] = []
        truncated = 0
        for start in tqdm(range(0, len(frame), batch_size), desc=f"{split}: tokenize pairs", unit="batch", dynamic_ncols=True):
            batch = frame.iloc[start : start + batch_size]
            encoded = tokenizer(
                batch["query_text"].astype(str).tolist(),
                batch["doc_text"].astype(str).tolist(),
                add_special_tokens=True,
                truncation="only_second",
                max_length=max_length,
                padding=False,
                return_attention_mask=True,
                return_token_type_ids=True,
            )
            doc_encoded = tokenizer(
                batch["doc_text"].astype(str).tolist(),
                add_special_tokens=False,
                truncation=False,
                padding=False,
            )["input_ids"]
            for offset, row in enumerate(batch.itertuples(index=False)):
                ids = encoded["input_ids"][offset]
                mask = encoded["attention_mask"][offset]
                if any(value == tokenizer.pad_token_id for value in ids) and tokenizer.pad_token_id is not None:
                    raise RuntimeError("Padding token found despite padding=False")
                # Count passage truncation against its original untruncated token length.
                doc_tokens = doc_encoded[offset]
                pair_budget = max_length - query_lengths[str(row.query_text)] - special_count
                was_truncated = len(doc_tokens) > pair_budget
                truncated += int(was_truncated)
                row_index = int(batch["_source_row"].iloc[offset])
                row_dict = row._asdict()
                records.append({
                    "pair_id": f"{split}:{row_index}:{row_dict['query_id']}:{row_dict['doc_id']}",
                    "split": split,
                    "row_index": row_index,
                    "query_id": str(row_dict["query_id"]),
                    "doc_id": str(row_dict["doc_id"]),
                    "input_ids": ids,
                    "attention_mask": mask,
                    "token_type_ids": encoded.get("token_type_ids", [[0] * len(ids) for _ in range(len(batch))])[offset],
                    "label": _optional(row_dict, "label"),
                    "is_judged": int(row_dict["is_judged"]),
                    "relevance_score": _optional(row_dict, "relevance_score"),
                    "negative_provenance": _optional(row_dict, "negative_provenance"),
                    "candidate_source": _optional(row_dict, "candidate_source"),
                    "candidate_rank": _optional(row_dict, "candidate_rank"),
                    "retrieval_score": _optional(row_dict, "retrieval_score"),
                    "seed": int(row_dict["seed"]),
                })
        split_dir = output_dir / split
        if split_dir.exists():
            import shutil
            shutil.rmtree(split_dir)
        if records:
            dataset = Dataset.from_list(records, features=TOKEN_FEATURES)
        else:
            dataset = Dataset.from_dict({name: [] for name in TOKEN_FEATURES}, features=TOKEN_FEATURES)
        dataset.save_to_disk(str(split_dir))
        denominator = len(frame)
        split_stats[split] = {
            "input_rows": len(valid_mask),
            "saved_rows": len(records),
            "rejected_overlong_queries": len(rejected),
            "passages_truncated": truncated,
            "passage_truncation_fraction": (truncated / denominator) if denominator else 0.0,
            "max_observed_sequence_length": max((len(row["input_ids"]) for row in records), default=0),
        }
        LOGGER.info(
            "%s saved %d/%d rows; passage truncation %d (%.2f%%); rejected overlong queries %d",
            split, len(records), len(valid_mask), truncated,
            100 * split_stats[split]["passage_truncation_fraction"], len(rejected),
        )

    exclusions_path = output_dir / "exclusions.jsonl"
    exclusions_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in exclusions), encoding="utf-8")
    manifest = {
        "model": {"name": model_name, "revision": revision, "local_dir": str(local_model_dir.resolve())},
        "tokenizer": {
            "max_length": max_length,
            "truncation": "only_second",
            "padding": False,
            "padding_side": tokenizer.padding_side,
            "tokenizer_declared_max_length": tokenizer_declared_max_length,
            "special_tokens_added_for_pair": special_count,
            "overlong_query_policy": "reject",
        },
        "seed": seed,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "splits": split_stats,
        "exclusions": {"path": exclusions_path.name, "rows": len(exclusions)},
        "package_versions": {name: _version(name) for name in ("transformers", "tokenizers", "datasets", "pyarrow", "pandas", "torch")},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["output_sha256"] = _file_hashes(output_dir)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
