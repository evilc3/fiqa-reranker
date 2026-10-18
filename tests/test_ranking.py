import pandas as pd

from fiqa_reranker.ranking import ranking_metrics


def test_ranking_metrics_uses_qrels_and_does_not_require_labels_on_candidates():
    qrels = {"q1": {"relevant": 1.0, "missed": 1.0}}
    predictions = pd.DataFrame(
        [
            {"query_id": "q1", "doc_id": "unknown", "score": 3.0},
            {"query_id": "q1", "doc_id": "relevant", "score": 2.0},
        ]
    )
    metrics = ranking_metrics(predictions, qrels)
    assert metrics["queries"] == 1
    assert metrics["mrr@10"] == 0.5
    assert 0 < metrics["ndcg@10"] < 1


def test_ranking_metrics_returns_zero_for_missing_candidate_list():
    metrics = ranking_metrics(pd.DataFrame(columns=["query_id", "doc_id", "score"]), {"q1": {"d1": 1.0}})
    assert metrics["queries"] == 1
    assert metrics["mrr@10"] == 0.0
    assert metrics["ndcg@10"] == 0.0
