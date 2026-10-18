import json

import pandas as pd
from datasets import load_from_disk
from transformers import BertTokenizerFast

from fiqa_reranker.tokenization import tokenize_splits


def _tiny_tokenizer(tmp_path):
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "what", "is", "a", "stock", "answer", "yes", "no"]
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("\n".join(vocab), encoding="utf-8")
    return BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=True)


def test_arrow_reload_alignment_labels_and_no_padding(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    rows = pd.DataFrame(
        [
            {"query_id": "q1", "doc_id": "d1", "query_text": "what is a stock", "doc_text": "answer yes", "label": 1, "is_judged": 1, "relevance_score": 1, "negative_provenance": None, "candidate_source": "qrels_positive", "candidate_rank": None, "retrieval_score": None, "seed": 42},
            {"query_id": "q2", "doc_id": "d2", "query_text": "what is a stock", "doc_text": "answer no", "label": None, "is_judged": 0, "relevance_score": None, "negative_provenance": None, "candidate_source": None, "candidate_rank": 1, "retrieval_score": 0.2, "seed": 42},
        ]
    )
    rows.iloc[[0]].to_csv(input_dir / "train.tsv", sep="\t", index=False)
    for split in ("validation", "test"):
        rows.to_csv(input_dir / f"{split}.tsv", sep="\t", index=False)
    output_dir = tmp_path / "arrow"
    tokenizer = _tiny_tokenizer(tmp_path)
    manifest = tokenize_splits(
        input_dir=input_dir,
        output_dir=output_dir,
        tokenizer=tokenizer,
        max_length=16,
        seed=42,
        model_name="test/local",
        revision="a" * 40,
        local_model_dir=tmp_path / "model",
        batch_size=1,
    )
    for split in ("train", "validation", "test"):
        dataset = load_from_disk(str(output_dir / split))
        assert dataset[0]["pair_id"] == f"{split}:0:q1:d1"
        for row in dataset:
            assert len(row["input_ids"]) == len(row["attention_mask"])
            assert len(row["input_ids"]) == len(row["token_type_ids"])
            assert tokenizer.pad_token_id not in row["input_ids"]
        assert dataset[0]["label"] == 1.0
        if split != "train":
            assert dataset[1]["label"] is None
            assert dataset[1]["is_judged"] == 0
    assert manifest["splits"]["validation"]["saved_rows"] == 2
    saved_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest["tokenizer"]["padding"] is False
    assert saved_manifest["output_sha256"]


def test_overlong_queries_are_logged_and_excluded(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    rows = pd.DataFrame(
        [{"query_id": "q1", "doc_id": "d1", "query_text": "what is a stock", "doc_text": "answer yes", "label": 1, "is_judged": 1, "seed": 42}]
    )
    for split in ("train", "validation", "test"):
        rows.to_csv(input_dir / f"{split}.tsv", sep="\t", index=False)
    output_dir = tmp_path / "arrow"
    manifest = tokenize_splits(
        input_dir=input_dir,
        output_dir=output_dir,
        tokenizer=_tiny_tokenizer(tmp_path),
        max_length=4,
        seed=42,
        model_name="test/local",
        revision="a" * 40,
        local_model_dir=tmp_path / "model",
        batch_size=1,
    )
    assert manifest["splits"]["train"]["saved_rows"] == 0
    assert manifest["splits"]["train"]["rejected_overlong_queries"] == 1
    excluded = (output_dir / "exclusions.jsonl").read_text(encoding="utf-8")
    assert '"reason": "overlong_query"' in excluded
