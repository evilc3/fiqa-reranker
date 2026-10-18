import pandas as pd
import pytest

from fiqa_reranker.data_pipeline import (
    BM25Index,
    DataValidationError,
    build_evaluation_candidates,
    build_training_examples,
    build_training_examples_fast,
    clean_text,
    export_pipeline,
    prepare_bundle,
)


def tiny_inputs():
    corpus = pd.DataFrame(
        [
            {"_id": "d1", "title": "", "text": "$ACME filed a 10-K with 5% growth."},
            {"_id": "d2", "title": "Loan", "text": "Variable interest rates."},
            {"_id": "d3", "title": "Bank", "text": "Savings account details."},
            {"_id": "d4", "title": "Markets", "text": "S&P 500 and ETF prices."},
            {"_id": "d5", "title": "Credit", "text": "APR and credit card balances."},
        ]
    )
    queries = pd.DataFrame(
        [
            {"_id": "q1", "title": "", "text": "What is ACME's 10-K growth?"},
            {"_id": "q2", "title": "", "text": "How do variable loan rates work?"},
            {"_id": "q3", "title": "", "text": "What is an ETF?"},
        ]
    )
    qrels = {
        "train": pd.DataFrame(
            [
                {"query-id":  "q1", "corpus-id": "d1", "score": 1},
                {"query-id":  "q1", "corpus-id": "d2", "score": 0},
                {"query-id":  "q1", "corpus-id": "d2", "score": 1},  # conflicting pair is excluded
                {"query-id":  "q1", "corpus-id": "missing-doc", "score": 1},
            ]
        ),
        "validation": pd.DataFrame([{"query-id": "q2", "corpus-id": "d2", "score": 1}]),
        "test": pd.DataFrame([{"query-id": "q3", "corpus-id": "d4", "score": 1}]),
    }
    return corpus, queries, qrels


def test_clean_text_preserves_financial_symbols_and_numbers():
    assert clean_text("  $ACME\t10-K  grew 5%. ") == "$ACME 10-K grew 5%."


def test_bm25_retrieval_is_deterministic_and_finds_financial_terms():
    corpus, _, _ = tiny_inputs()
    from fiqa_reranker.data_pipeline import _prepare_entity

    prepared, _ = _prepare_entity(corpus, "corpus", [])
    index = BM25Index(prepared)
    first = index.search("$ACME 10-K", top_k=3)
    second = index.search("$ACME 10-K", top_k=3)
    assert first == second
    assert first[0]["doc_id"] == "d1"
    assert first[0]["retrieval_score"] > 0


def test_bm25_partial_top_k_matches_full_sort_including_cutoff_ties():
    corpus = pd.DataFrame(
        [
            {"_id": "d4", "title": "", "text": "credit report"},
            {"_id": "d2", "title": "", "text": "credit report"},
            {"_id": "d1", "title": "", "text": "credit report"},
            {"_id": "d3", "title": "", "text": "credit credit report"},
            {"_id": "d5", "title": "", "text": "loan"},
        ]
    )
    from fiqa_reranker.data_pipeline import _prepare_entity

    prepared, _ = _prepare_entity(corpus, "corpus", [])
    index = BM25Index(prepared)
    assert index.search("credit", top_k=2) == index.search("credit", top_k=2, partial_top_k=True)


def test_validation_logs_conflicts_and_bad_joins():
    corpus, queries, qrels = tiny_inputs()
    bundle = prepare_bundle(corpus, queries, qrels)
    reasons = [item["reason"] for item in bundle.exclusions]
    assert "conflicting_judgments" in reasons
    assert "document_id_not_found" in reasons
    assert not ((bundle.qrels["train"].query_id == "q1") & (bundle.qrels["train"].doc_id == "d2")).any()
    assert bundle.qrels["train"].query_id.tolist() == ["q1"]


def test_duplicate_ids_are_dropped_or_excluded_and_empty_text_logged():
    corpus, queries, qrels = tiny_inputs()
    corpus = pd.concat(
        [
            corpus,
            pd.DataFrame(
                [
                    {"_id": "d5", "title": "Credit", "text": "APR and credit card balances."},
                    {"_id": "d4", "title": "Changed", "text": "Conflicting same ID."},
                    {"_id": "empty", "title": "", "text": "  "},
                    {"_id": "", "title": "No ID", "text": "Cannot join."},
                ]
            ),
        ],
        ignore_index=True,
    )
    bundle = prepare_bundle(corpus, queries, qrels)
    assert "d5" in set(bundle.corpus["_id"])
    assert not bundle.corpus["_id"].duplicated().any()
    assert "d4" not in set(bundle.corpus["_id"])  # conflicting content rows are all removed
    assert "empty_text" in {row["reason"] for row in bundle.exclusions}
    assert "missing_id" in {row["reason"] for row in bundle.exclusions}
    assert "duplicate_id_identical" in {row["reason"] for row in bundle.exclusions}


def test_query_leakage_across_official_splits_is_rejected():
    corpus, queries, qrels = tiny_inputs()
    qrels["test"] = pd.DataFrame([{"query-id": "q1", "corpus-id": "d1", "score": 1}])
    with pytest.raises(DataValidationError, match="Query leakage"):
        prepare_bundle(corpus, queries, qrels)


def test_duplicate_query_text_across_splits_is_rejected():
    corpus, queries, qrels = tiny_inputs()
    queries.loc[queries["_id"] == "q3", "text"] = queries.loc[queries["_id"] == "q1", "text"].iloc[0]
    with pytest.raises(DataValidationError, match="Duplicate query text"):
        prepare_bundle(corpus, queries, qrels)


def test_training_ratio_seed_and_positive_exclusion():
    corpus, queries, qrels = tiny_inputs()
    bundle = prepare_bundle(corpus, queries, qrels)
    retrieved = {
        "q1": [
            {"doc_id": "d1", "candidate_rank": 1, "retrieval_score": 10.0},
            {"doc_id": "d2", "candidate_rank": 2, "retrieval_score": 9.0},
            {"doc_id": "d3", "candidate_rank": 3, "retrieval_score": 8.0},
        ]
    }
    first = build_training_examples(bundle, retrieved, seed=73, negatives_per_positive=4, hard_negative_fraction=0.5)
    second = build_training_examples(bundle, retrieved, seed=73, negatives_per_positive=4, hard_negative_fraction=0.5)
    fast = build_training_examples_fast(bundle, retrieved, seed=73, negatives_per_positive=4, hard_negative_fraction=0.5)
    fast_repeat = build_training_examples_fast(bundle, retrieved, seed=73, negatives_per_positive=4, hard_negative_fraction=0.5)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(fast, fast_repeat)
    assert fast.label.value_counts().to_dict() == {0: 4, 1: 1}
    assert "d1" not in set(fast.loc[fast.label == 0, "doc_id"])
    assert first.label.value_counts().to_dict() == {0: 4, 1: 1}
    assert "d1" not in set(first.loc[first.label == 0, "doc_id"])
    hard = first[first.candidate_source == "bm25_hard_negative"]
    assert len(hard) == 2
    assert set(hard.negative_provenance) == {"retrieved_unjudged_false_negative_risk"}
    assert first.seed.eq(73).all()


def test_fast_sampler_excludes_all_known_positives_for_multi_positive_query():
    corpus, queries, qrels = tiny_inputs()
    qrels["train"] = pd.DataFrame(
        [
            {"query-id": "q1", "corpus-id": "d1", "score": 1},
            {"query-id": "q1", "corpus-id": "d2", "score": 1},
        ]
    )
    bundle = prepare_bundle(corpus, queries, qrels)
    result = build_training_examples_fast(
        bundle, {}, seed=29, negatives_per_positive=2, hard_negative_fraction=0.0,
    )
    positives = set(bundle.qrels["train"].loc[bundle.qrels["train"].score > 0, "doc_id"])
    negatives = set(result.loc[result.label == 0, "doc_id"])
    assert positives == {"d1", "d2"}
    assert not positives.intersection(negatives)
    assert result.label.value_counts().to_dict() == {0: 4, 1: 2}


def test_evaluation_candidates_are_not_balanced_and_unjudged_stay_unknown():
    corpus, queries, qrels = tiny_inputs()
    bundle = prepare_bundle(corpus, queries, qrels)
    retrieved = {
        "q2": [
            {"doc_id": "d3", "candidate_rank": 1, "retrieval_score": 3.0},
            {"doc_id": "d2", "candidate_rank": 2, "retrieval_score": 2.0},
        ]
    }
    candidates = build_evaluation_candidates(bundle, "validation", retrieved, seed=42)
    assert len(candidates) == 2
    assert candidates.label.tolist()[0] is None or pd.isna(candidates.label.tolist()[0])
    assert candidates.label.tolist()[1] == 1
    assert candidates.is_judged.tolist() == [0, 1]
    assert candidates.negative_provenance.eq("").all()
    assert candidates.candidate_rank.tolist() == [1, 2]


def test_export_writes_tsvs_exclusions_and_manifest(tmp_path):
    corpus, queries, qrels = tiny_inputs()
    bundle = prepare_bundle(corpus, queries, qrels)
    manifest = export_pipeline(
        bundle,
        out_dir=tmp_path,
        seed=12,
        retrieval_depth=4,
        eval_top_k=2,
        bm25_k1=1.2,
        bm25_b=0.75,
        negatives_per_positive=2,
        hard_negative_fraction=0.5,
        dataset_revisions={"test": "synthetic"},
        fingerprints={"test": "synthetic"},
        config={"test": True},
    )
    assert {"train.tsv", "validation.tsv", "test.tsv", "exclusions.tsv", "manifest.json"} <= {
        path.name for path in tmp_path.iterdir()
    }
    train = pd.read_csv(tmp_path / "train.tsv", sep="\t")
    assert train.label.value_counts().to_dict() == {0: 2, 1: 1}
    assert manifest["seed"] == 12
    assert manifest["counts"]["examples"]["test"] == len(pd.read_csv(tmp_path / "test.tsv", sep="\t"))

    fast_dir = tmp_path / "fast"
    fast_manifest = export_pipeline(
        bundle,
        out_dir=fast_dir,
        seed=12,
        retrieval_depth=4,
        eval_top_k=2,
        bm25_k1=1.2,
        bm25_b=0.75,
        negatives_per_positive=2,
        hard_negative_fraction=0.5,
        dataset_revisions={"test": "synthetic"},
        fingerprints={"test": "synthetic"},
        config={"test": True},
        fast_mode=True,
    )
    fast_train = pd.read_csv(fast_dir / "train.tsv", sep="\t")
    fast_test = pd.read_csv(fast_dir / "test.tsv", sep="\t")
    standard_test = pd.read_csv(tmp_path / "test.tsv", sep="\t")
    assert fast_train.columns.tolist() == train.columns.tolist()
    assert fast_train.query_text.notna().all() and fast_train.doc_text.notna().all()
    assert fast_test.query_text.notna().all() and fast_test.doc_text.notna().all()
    query_text = bundle.queries.set_index("_id")["combined_text"].to_dict()
    document_text = bundle.corpus.set_index("_id")["combined_text"].to_dict()
    assert fast_train.query_text.tolist() == fast_train.query_id.map(query_text).tolist()
    assert fast_train.doc_text.tolist() == fast_train.doc_id.map(document_text).tolist()
    pd.testing.assert_frame_equal(fast_test, standard_test)
    fast_positives = set(zip(fast_train.loc[fast_train.label == 1, "query_id"], fast_train.loc[fast_train.label == 1, "doc_id"]))
    standard_positives = set(zip(train.loc[train.label == 1, "query_id"], train.loc[train.label == 1, "doc_id"]))
    assert fast_positives == standard_positives
    assert fast_manifest["pipeline_variant"]["name"] == "fast"
