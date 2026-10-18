"""Easy-only FiQA training pairs with unchanged evaluation candidate files."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from fiqa_reranker.data_pipeline import (
    OUTPUT_COLUMNS,
    DataValidationError,
    PreparedBundle,
    _base_row,
    _sample_eligible_without_replacement,
    _stable_rng,
    _write_pair_tsv_in_chunks,
    write_exclusions,
)

LOGGER = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_easy_training_examples(
    bundle: PreparedBundle, *, seed: int, negatives_per_positive: int = 4,
) -> pd.DataFrame:
    """Sample unique negatives per query, independently of query/corpus order.

    Uniform corpus sampling does not establish topic difference or irrelevance.
    Sampled unjudged pairs receive training label zero with explicit provenance.
    """
    if negatives_per_positive < 1:
        raise ValueError("negatives_per_positive must be at least 1")
    if "train" not in bundle.qrels:
        raise DataValidationError("Missing official train qrels")
    corpus_ids = np.array(sorted(bundle.corpus["_id"].astype(str).unique()), dtype=object)
    corpus_set = set(corpus_ids)
    all_qrels = pd.concat(bundle.qrels.values(), ignore_index=True)
    known_positives = {
        str(qid): set(group.loc[group.score > 0, "doc_id"].astype(str))
        for qid, group in all_qrels.groupby("query_id", sort=True)
    }
    judgments = {
        (str(row.query_id), str(row.doc_id)): float(row.score)
        for row in all_qrels.itertuples(index=False)
    }
    positives = bundle.qrels["train"].loc[lambda frame: frame.score > 0].copy()
    positives["query_id"] = positives.query_id.astype(str)
    positives["doc_id"] = positives.doc_id.astype(str)
    positives = positives.sort_values(["query_id", "doc_id"], kind="stable")
    if positives.duplicated(["query_id", "doc_id"]).any():
        raise DataValidationError("Duplicate positive judgments; validate inputs first")
    records: list[dict[str, Any]] = []
    groups = positives.groupby("query_id", sort=True)
    for qid, group in tqdm(groups, total=groups.ngroups, desc="Easy negatives per query", unit="query", dynamic_ncols=True):
        excluded = known_positives.get(qid, set()) & corpus_set
        requested = len(group) * negatives_per_positive
        if requested > len(corpus_ids) - len(excluded):
            raise DataValidationError(
                f"Query {qid}: need {requested} unique negatives but only "
                f"{len(corpus_ids) - len(excluded)} eligible documents exist"
            )
        chosen = _sample_eligible_without_replacement(
            _stable_rng(seed, qid, "easy_only_v1"), corpus_ids, excluded, requested,
        )
        for position, positive in enumerate(group.itertuples(index=False)):
            records.append(_base_row(
                "train", qid, positive.doc_id, "", "", seed,
                label=1, relevance_score=float(positive.score), is_judged=True,
                negative_provenance="", candidate_source="qrels_positive",
            ))
            start = position * negatives_per_positive
            for did in chosen[start : start + negatives_per_positive]:
                score = judgments.get((qid, did))
                records.append(_base_row(
                    "train", qid, did, "", "", seed,
                    label=0, relevance_score=score, is_judged=score is not None,
                    negative_provenance=(
                        "random_judged_nonrelevant" if score is not None
                        else "random_unjudged_false_negative_risk"
                    ),
                    candidate_source="random_easy_negative",
                ))
    result = pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)
    if result.duplicated(["query_id", "doc_id"]).any():
        raise DataValidationError("Duplicate training pairs detected")
    if (result.label == 0).sum() != (result.label == 1).sum() * negatives_per_positive:
        raise DataValidationError("Training ratio does not match configuration")
    return result


def verify_evaluation_source(source: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Check source provenance and hashes before copying any evaluation data."""
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    dataset = config["dataset"]
    expected = {
        "corpus_and_queries": f"{dataset['name']}@{dataset['revision']}",
        "qrels": f"{dataset['qrels']['name']}@{dataset['qrels']['revision']}",
    }
    for key, value in expected.items():
        recorded = manifest["dataset"][key]
        name, revision = value.rsplit("@", 1)
        recorded_name, recorded_revision = recorded.rsplit("@", 1)
        if name != recorded_name or not (
            revision.startswith(recorded_revision) or recorded_revision.startswith(revision)
        ):
            raise DataValidationError(f"Evaluation dataset revision mismatch: {key}")
    for split in ("validation", "test"):
        if manifest["artifacts"]["files"][split] != f"{split}.tsv":
            raise DataValidationError(f"Unexpected source filename for {split}")
        if sha256_file(source / f"{split}.tsv") != manifest["artifacts"]["sha256"][split]:
            raise DataValidationError(f"Evaluation source checksum mismatch: {split}")
    return manifest


def export_easy_pipeline(
    bundle: PreparedBundle, *, config: dict[str, Any], fingerprints: dict[str, str],
    source_dir: Path, output_dir: Path,
) -> dict[str, Any]:
    """Export training pairs plus byte-identical validation/test TSV copies."""
    source_dir, output_dir = source_dir.resolve(), output_dir.resolve()
    if source_dir == output_dir or source_dir in output_dir.parents or output_dir in source_dir.parents:
        raise ValueError("Evaluation source and output must be separate, non-overlapping directories")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}; choose a new --output-dir")
    LOGGER.info("Stage 2/5: verify original evaluation provenance and hashes")
    original = verify_evaluation_source(source_dir, config)
    LOGGER.info("Stage 3/5: sample unique easy negatives; no BM25 training mining")
    train = build_easy_training_examples(
        bundle, seed=int(config["seed"]),
        negatives_per_positive=int(config["sampling"]["negatives_per_positive"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Stage 4/5: export train and copy unchanged evaluation TSVs")
    _write_pair_tsv_in_chunks(train, output_dir / "train.tsv", bundle)
    for split in tqdm(("validation", "test"), desc="Copy evaluation files", unit="file", dynamic_ncols=True):
        shutil.copyfile(source_dir / f"{split}.tsv", output_dir / f"{split}.tsv")
        if sha256_file(output_dir / f"{split}.tsv") != original["artifacts"]["sha256"][split]:
            raise DataValidationError(f"Copied evaluation checksum mismatch: {split}")
    write_exclusions(bundle.exclusions, output_dir / "exclusions.tsv")
    files = {key: f"{key}.tsv" for key in ("train", "validation", "test", "exclusions")}
    LOGGER.info("Stage 5/5: save counts, revisions, configuration and checksums")
    manifest = {
        "dataset": original["dataset"],
        "dataset_fingerprints": fingerprints,
        "configuration": config,
        "seed": config["seed"],
        "pipeline_variant": {"name": "easy_only_v1", "unique_negatives_per_query": True},
        "negative_policy": {
            "positive_definition": "official train qrels score > 0",
            "negatives_per_positive": config["sampling"]["negatives_per_positive"],
            "hard_negative_fraction": 0.0,
            "sampling": "sorted corpus IDs, SHA256-derived per-query NumPy RNG, uniform rejection sampling without replacement",
            "exclusion": "all known positive document IDs for the query across official qrels",
            "false_negative_risk": "Random unjudged passages are assumed negative for training; topic difference and irrelevance are not guaranteed.",
        },
        "evaluation_source": {
            "directory": str(source_dir),
            "manifest_sha256": sha256_file(source_dir / "manifest.json"),
            "policy": "byte-identical copies; original candidate order, labels, ranks, texts and seeds preserved",
        },
        "counts": {
            "examples": {"train": len(train), **{s: original["counts"]["examples"][s] for s in ("validation", "test")}},
            "train_positives": int((train.label == 1).sum()),
            "train_negatives": int((train.label == 0).sum()),
            "train_duplicate_pairs": int(train.duplicated(["query_id", "doc_id"]).sum()),
            "exclusions": len(bundle.exclusions),
        },
        "artifacts": {"files": files, "sha256": {key: sha256_file(output_dir / value) for key, value in files.items()}},
        "software": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "datasets")},
        "limitations": [
            "This variant changes both negative source and repeated-pair weighting; differences cannot be attributed to negative source alone.",
            "Use validation for model selection; repeated test inspection must not guide tuning.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
