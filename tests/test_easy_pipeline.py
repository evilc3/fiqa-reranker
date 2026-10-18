"""Small synthetic checks; no downloads, training, or real dataset generation."""

import json

import pandas as pd
import pytest

from fiqa_reranker.data_pipeline import DataValidationError, PreparedBundle
from fiqa_reranker.easy_pipeline import (
    build_easy_training_examples,
    export_easy_pipeline,
    sha256_file,
    verify_evaluation_source,
)


def fixture_bundle():
    return PreparedBundle(
        corpus=pd.DataFrame([{"_id": f"d{i}", "combined_text": f"Passage ${i} 5%."} for i in range(40)]),
        queries=pd.DataFrame([{"_id": qid, "combined_text": f"Question {qid}?"} for qid in ("q1", "q2", "q3")]),
        qrels={
            "train": pd.DataFrame([
                {"query_id": "q1", "doc_id": "d0", "score": 1},
                {"query_id": "q1", "doc_id": "d1", "score": 1},
                {"query_id": "q2", "doc_id": "d2", "score": 1},
            ]),
            "validation": pd.DataFrame([{"query_id": "q3", "doc_id": "d3", "score": 1}]),
            # Exercises exclusion over all supplied qrels, not only train.
            "additional_judgments": pd.DataFrame([{"query_id": "q1", "doc_id": "d4", "score": 1}]),
        },
    )


def test_ratio_known_positive_exclusion_and_unique_pairs():
    bundle = fixture_bundle()
    pairs = build_easy_training_examples(bundle, seed=42)
    assert len(pairs) == 15
    assert (pairs.label == 1).sum() == 3
    assert (pairs.label == 0).sum() == 12
    assert not pairs.duplicated(["query_id", "doc_id"]).any()
    q1_negatives = set(pairs.loc[(pairs.query_id == "q1") & (pairs.label == 0), "doc_id"])
    assert len(q1_negatives) == 8
    assert q1_negatives.isdisjoint({"d0", "d1", "d4"})
    negatives = pairs[pairs.label == 0]
    assert negatives.candidate_source.eq("random_easy_negative").all()
    assert negatives.negative_provenance.eq("random_unjudged_false_negative_risk").all()
    assert negatives.candidate_rank.isna().all()


def test_sampling_independent_of_input_order_and_reproducible():
    bundle = fixture_bundle()
    expected = build_easy_training_examples(bundle, seed=42)
    bundle.corpus = bundle.corpus.sample(frac=1, random_state=1)
    bundle.qrels = {key: frame.iloc[::-1] for key, frame in reversed(list(bundle.qrels.items()))}
    pd.testing.assert_frame_equal(expected, build_easy_training_examples(bundle, seed=42))
    assert not expected.equals(build_easy_training_examples(bundle, seed=99))


def test_insufficient_unique_negatives_raises_instead_of_changing_ratio():
    bundle = fixture_bundle()
    bundle.corpus = bundle.corpus.iloc[:5]
    with pytest.raises(DataValidationError, match="unique negatives"):
        build_easy_training_examples(bundle, seed=42)


def evaluation_fixture(tmp_path):
    source = tmp_path / "original"
    source.mkdir()
    for split in ("validation", "test"):
        (source / f"{split}.tsv").write_bytes(
            b"split\tquery_id\tdoc_id\tlabel\tis_judged\n" + f"{split}\tq3\td3\t\t0\n".encode()
        )
    manifest = {
        "dataset": {"corpus_and_queries": "BeIR/fiqa@abc123", "qrels": "BeIR/fiqa-qrels@def456"},
        "artifacts": {
            "files": {split: f"{split}.tsv" for split in ("validation", "test")},
            "sha256": {split: sha256_file(source / f"{split}.tsv") for split in ("validation", "test")},
        },
        "counts": {"examples": {"validation": 1, "test": 1}},
    }
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = {"seed": 42, "sampling": {"negatives_per_positive": 4}, "dataset": {
        "name": "BeIR/fiqa", "revision": "abc123", "qrels": {"name": "BeIR/fiqa-qrels", "revision": "def456"},
    }}
    return source, config


def test_export_preserves_evaluation_bytes_and_restores_train_text(tmp_path):
    source, config = evaluation_fixture(tmp_path)
    out = tmp_path / "easy"
    report = export_easy_pipeline(fixture_bundle(), config=config, fingerprints={}, source_dir=source, output_dir=out)
    for split in ("validation", "test"):
        assert (source / f"{split}.tsv").read_bytes() == (out / f"{split}.tsv").read_bytes()
    train = pd.read_csv(out / "train.tsv", sep="\t")
    assert train.query_text.notna().all()
    assert train.doc_text.str.contains("Passage ", regex=False).all()
    assert report["counts"]["train_duplicate_pairs"] == 0
    with pytest.raises(FileExistsError):
        export_easy_pipeline(fixture_bundle(), config=config, fingerprints={}, source_dir=source, output_dir=out)


def test_source_hash_or_revision_mismatch_rejected(tmp_path):
    source, config = evaluation_fixture(tmp_path)
    config["dataset"]["revision"] = "different"
    with pytest.raises(DataValidationError, match="revision mismatch"):
        verify_evaluation_source(source, config)
    config["dataset"]["revision"] = "abc123"
    (source / "validation.tsv").write_text("modified", encoding="utf-8")
    with pytest.raises(DataValidationError, match="checksum mismatch"):
        verify_evaluation_source(source, config)
