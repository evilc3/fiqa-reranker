"""Reproducible alignment, label, negative, and truncation audits for FiQA pairs."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset, load_from_disk
from tqdm.auto import tqdm
from transformers import AutoTokenizer


def _is_null(value: Any) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value))


def _text(value: Any) -> str:
    return "" if _is_null(value) else str(value)


def _sample(items: list[int], count: int, rng: random.Random) -> list[int]:
    return rng.sample(items, min(count, len(items))) if items else []


def _reservoir_add(
    reservoirs: dict[str, list[dict[str, Any]]],
    seen: Counter[str],
    group: str,
    row: dict[str, Any],
    limit: int,
    rng: random.Random,
) -> None:
    """Keep a fixed-size reproducible sample without retaining every long text."""
    seen[group] += 1
    reservoir = reservoirs[group]
    if len(reservoir) < limit:
        reservoir.append(row)
    else:
        replacement = rng.randrange(seen[group])
        if replacement < limit:
            reservoir[replacement] = row


def _decode_segments(tokenizer, input_ids: list[int], token_type_ids: list[int]) -> tuple[str, str]:
    query = [token for token, segment in zip(input_ids, token_type_ids, strict=True) if segment == 0]
    passage = [token for token, segment in zip(input_ids, token_type_ids, strict=True) if segment == 1]
    return (
        tokenizer.decode(query, skip_special_tokens=True),
        tokenizer.decode(passage, skip_special_tokens=True),
    )


def _negative_category(split: str, row: dict[str, Any]) -> str:
    if split == "train":
        source = _text(row.get("candidate_source"))
        provenance = _text(row.get("negative_provenance"))
        if source == "random_fallback" or provenance.startswith("random_fallback"):
            return "fallback_negative"
        if source in {"bm25_hard_negative", "retrieved_hard_negative"} or provenance.startswith("retrieved_"):
            return "hard_negative"
        if source in {"random_easy_negative", "random_fallback"} or provenance.startswith("random_"):
            return "easy_negative"
        return "other_labeled_negative"
    if int(row.get("is_judged", 0)) == 0:
        return "unjudged_candidate"
    if float(row.get("label") or 0) == 0:
        return "judged_nonrelevant"
    return "positive_candidate"


def audit_fiqa_data(
    *,
    tokenized_dir: Path,
    processed_dir: Path,
    model_dir: Path,
    qrels_name: str,
    qrels_revision: str,
    qrels_splits: dict[str, str],
    output_dir: Path,
    seed: int = 42,
    examples_per_category: int = 30,
    batch_size: int = 512,
) -> dict[str, Any]:
    """Audit every split, writing machine-readable findings and review samples."""
    if examples_per_category < 1 or batch_size < 1:
        raise ValueError("examples_per_category and batch_size must be positive")
    manifest_path = tokenized_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    max_length = int(manifest["tokenizer"]["max_length"])
    if manifest["tokenizer"]["truncation"] != "only_second" or manifest["tokenizer"]["padding"] is not False:
        raise ValueError("Unexpected tokenization settings; expected truncation=only_second and padding=false")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokenizer.model_max_length = max(int(tokenizer.model_max_length), 1_000_000)
    special_count = tokenizer.num_special_tokens_to_add(pair=True)

    qrels_by_split: dict[str, dict[str, dict[str, float]]] = {}
    qrels_all: dict[str, dict[str, float]] = defaultdict(dict)
    print("Stage 1/4: loading pinned official qrels", flush=True)
    for split, source_split in qrels_splits.items():
        frame = load_dataset(qrels_name, split=source_split, revision=qrels_revision).to_pandas()
        needed = {"query-id", "corpus-id", "score"}
        if not needed.issubset(frame.columns):
            raise ValueError(f"Qrels {source_split} lacks columns {sorted(needed - set(frame.columns))}")
        mapping: dict[str, dict[str, float]] = defaultdict(dict)
        for qid_value, did_value, score_value in frame[["query-id", "corpus-id", "score"]].itertuples(index=False, name=None):
            qid, did, score = str(qid_value), str(did_value), float(score_value)
            mapping[qid][did] = score
            if did in qrels_all[qid] and qrels_all[qid][did] != score:
                raise ValueError(f"Conflicting official qrels for query={qid}, doc={did}")
            qrels_all[qid][did] = score
        qrels_by_split[split] = dict(mapping)

    query_owners: dict[str, str] = {}
    split_overlap: list[dict[str, str]] = []
    for split, qrels in qrels_by_split.items():
        for qid in qrels:
            prior = query_owners.setdefault(qid, split)
            if prior != split:
                split_overlap.append({"query_id": qid, "first_split": prior, "second_split": split})

    output_dir.mkdir(parents=True, exist_ok=True)
    alignment_issues: list[dict[str, Any]] = []
    label_issues: list[dict[str, Any]] = []
    pair_duplicate_rows: list[dict[str, Any]] = []
    split_reports: dict[str, Any] = {}
    review_reservoirs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    alignment_reservoirs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    truncation_reservoirs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    review_seen: Counter[str] = Counter()
    alignment_seen: Counter[str] = Counter()
    truncation_seen: Counter[str] = Counter()
    rng = random.Random(seed)

    for split in ("train", "validation", "test"):
        print(f"Stage 2/4: source/token alignment and label checks: {split}", flush=True)
        source_path = processed_dir / f"{split}.tsv"
        source = pd.read_csv(source_path, sep="\t", dtype={"query_id": "string", "doc_id": "string", "query_text": "string", "doc_text": "string"}, keep_default_na=True)
        ds = load_from_disk(str(tokenized_dir / split))
        if len(source) != len(ds):
            alignment_issues.append({"split": split, "row_index": None, "issue": "source_tokenized_row_count_mismatch", "source_rows": len(source), "tokenized_rows": len(ds)})

        ids_seen: set[str] = set()
        truncated_counts: Counter[tuple[str, str]] = Counter()
        total_counts: Counter[tuple[str, str]] = Counter()
        row_indices_seen: set[int] = set()
        split_positive_set = {qid: {did for did, score in docs.items() if score > 0} for qid, docs in qrels_all.items()}
        pair_first_row: dict[tuple[str, str], int] = {}

        for start in tqdm(range(0, len(ds), batch_size), desc=f"{split}: re-tokenize and align", unit="batch", dynamic_ncols=True):
            batch = ds[start : start + batch_size]
            row_indices = [int(i) for i in batch["row_index"]]
            source_rows: list[pd.Series | None] = []
            for local_i, row_index in enumerate(row_indices):
                absolute_i = start + local_i
                if row_index < 0 or row_index >= len(source):
                    source_rows.append(None)
                    alignment_issues.append({"split": split, "row_index": row_index, "pair_id": batch["pair_id"][local_i], "issue": "source_row_index_out_of_range"})
                    continue
                source_row = source.iloc[row_index]
                source_rows.append(source_row)
                if row_index in row_indices_seen:
                    alignment_issues.append({"split": split, "row_index": row_index, "pair_id": batch["pair_id"][local_i], "issue": "duplicate_source_row_index"})
                row_indices_seen.add(row_index)
                qid, did = str(batch["query_id"][local_i]), str(batch["doc_id"][local_i])
                expected_pair_id = f"{split}:{row_index}:{qid}:{did}"
                if str(batch["pair_id"][local_i]) != expected_pair_id:
                    alignment_issues.append({"split": split, "row_index": row_index, "pair_id": batch["pair_id"][local_i], "issue": "pair_id_mismatch", "expected": expected_pair_id})
                if (qid, did) in pair_first_row:
                    pair_duplicate_rows.append({
                        "split": split,
                        "first_row_index": pair_first_row[(qid, did)],
                        "duplicate_row_index": row_index,
                        "query_id": qid,
                        "doc_id": did,
                        "label": batch["label"][local_i],
                        "negative_provenance": _text(batch["negative_provenance"][local_i]),
                        "candidate_source": _text(batch["candidate_source"][local_i]),
                    })
                else:
                    pair_first_row[(qid, did)] = row_index
                if qid != str(source_row["query_id"]) or did != str(source_row["doc_id"]):
                    alignment_issues.append({"split": split, "row_index": row_index, "query_id": qid, "doc_id": did, "issue": "source_id_mismatch", "source_query_id": str(source_row["query_id"]), "source_doc_id": str(source_row["doc_id"])})
                label = batch["label"][local_i]
                label = None if _is_null(label) else float(label)
                score = batch["relevance_score"][local_i]
                score = None if _is_null(score) else float(score)
                judged = int(batch["is_judged"][local_i])
                qrel_score = qrels_all.get(qid, {}).get(did)
                issue_base = {"split": split, "row_index": row_index, "pair_id": str(batch["pair_id"][local_i]), "query_id": qid, "doc_id": did, "label": label, "relevance_score": score, "qrels_score": qrel_score}
                if label not in (None, 0.0, 1.0):
                    label_issues.append({**issue_base, "issue": "non_binary_label"})
                if split == "train":
                    if label is None:
                        label_issues.append({**issue_base, "issue": "training_label_is_null"})
                    if (qrel_score is None) != (judged == 0):
                        label_issues.append({**issue_base, "issue": "is_judged_disagrees_with_qrels"})
                    if qrel_score is not None and score != qrel_score:
                        label_issues.append({**issue_base, "issue": "relevance_score_disagrees_with_qrels"})
                    if qrel_score is not None and label != float(qrel_score > 0):
                        label_issues.append({**issue_base, "issue": "label_disagrees_with_qrels_relevance"})
                    if label == 1.0 and (qrel_score is None or qrel_score <= 0):
                        label_issues.append({**issue_base, "issue": "positive_missing_from_positive_qrels"})
                    if label == 0.0 and did in split_positive_set.get(qid, set()):
                        label_issues.append({**issue_base, "issue": "negative_is_known_positive"})
                    if label == 1.0 and (score is None or score <= 0):
                        label_issues.append({**issue_base, "issue": "positive_relevance_score_not_positive"})
                else:
                    if qrel_score is not None and qrel_score > 0 and label != 1.0:
                        label_issues.append({**issue_base, "issue": "qrels_positive_not_labeled_positive"})
                    if qrel_score is None and label is not None:
                        label_issues.append({**issue_base, "issue": "unjudged_candidate_has_invented_label"})
                    if (qrel_score is None) != (judged == 0):
                        label_issues.append({**issue_base, "issue": "is_judged_disagrees_with_qrels"})
                    if qrel_score is not None and score != qrel_score:
                        label_issues.append({**issue_base, "issue": "relevance_score_disagrees_with_qrels"})

                category = "positive" if label == 1.0 else "negative" if label == 0.0 else "unjudged"
                provenance = _text(batch["negative_provenance"][local_i]) or "none"
                total_counts[(category, provenance)] += 1
                # Full truncation and exact sequence comparison are computed in batches below.
            if not all(row is not None for row in source_rows):
                # Cannot re-tokenize rows with broken source alignment; continue auditing the rest.
                continue
            queries = [str(row["query_text"]) for row in source_rows]
            passages = [str(row["doc_text"]) for row in source_rows]
            expected = tokenizer(
                queries,
                passages,
                add_special_tokens=True,
                truncation="only_second",
                max_length=max_length,
                padding=False,
                return_attention_mask=True,
                return_token_type_ids=True,
            )
            query_enc = tokenizer(queries, add_special_tokens=False, truncation=False, padding=False)["input_ids"]
            passage_enc = tokenizer(passages, add_special_tokens=False, truncation=False, padding=False)["input_ids"]
            for i, row in enumerate(source_rows):
                row_index = row_indices[i]
                qid, did = str(batch["query_id"][i]), str(batch["doc_id"][i])
                saved_ids = [int(x) for x in batch["input_ids"][i]]
                saved_mask = [int(x) for x in batch["attention_mask"][i]]
                saved_types = [int(x) for x in batch["token_type_ids"][i]]
                mismatches = []
                if saved_ids != [int(x) for x in expected["input_ids"][i]]:
                    mismatches.append("input_ids")
                if saved_mask != [int(x) for x in expected["attention_mask"][i]]:
                    mismatches.append("attention_mask")
                expected_types = expected.get("token_type_ids", [[0] * len(ids) for ids in expected["input_ids"]])[i]
                if saved_types != [int(x) for x in expected_types]:
                    mismatches.append("token_type_ids")
                if len(saved_ids) != len(saved_mask) or len(saved_ids) != len(saved_types):
                    mismatches.append("array_length_alignment")
                if tokenizer.pad_token_id is not None and tokenizer.pad_token_id in saved_ids:
                    mismatches.append("unexpected_padding_token")
                if mismatches:
                    alignment_issues.append({"split": split, "row_index": row_index, "pair_id": batch["pair_id"][i], "query_id": qid, "doc_id": did, "issue": "tokenization_mismatch", "fields": ",".join(mismatches)})

                label = batch["label"][i]
                label = None if _is_null(label) else float(label)
                label_class = "positive" if label == 1.0 else "negative" if label == 0.0 else "unjudged"
                query_budget = max_length - len(query_enc[i]) - special_count
                was_truncated = len(passage_enc[i]) > query_budget
                truncated_counts[(label_class, _text(batch["negative_provenance"][i]) or "none")] += int(was_truncated)
                decoded_query, decoded_passage = _decode_segments(tokenizer, saved_ids, saved_types)
                review_row = {
                    "split": split,
                    "row_index": row_index,
                    "pair_id": str(batch["pair_id"][i]),
                    "query_id": qid,
                    "doc_id": did,
                    "label": label,
                    "is_judged": int(batch["is_judged"][i]),
                    "relevance_score": None if _is_null(batch["relevance_score"][i]) else float(batch["relevance_score"][i]),
                    "negative_provenance": _text(batch["negative_provenance"][i]),
                    "candidate_source": _text(batch["candidate_source"][i]),
                    "candidate_rank": batch["candidate_rank"][i],
                    "retrieval_score": batch["retrieval_score"][i],
                    "truncated": was_truncated,
                    "query_text": str(row["query_text"]),
                    "full_passage_text": str(row["doc_text"]),
                    "decoded_query": decoded_query,
                    "retained_passage_text": decoded_passage,
                    "full_passage_tokens": len(passage_enc[i]),
                    "retained_passage_tokens": min(len(passage_enc[i]), max(0, query_budget)),
                    "sequence_tokens": len(saved_ids),
                    "tokenization_matches": not mismatches,
                }
                if label == 0.0 or (split != "train" and label is None):
                    review_row["review_category"] = _negative_category(split, review_row)
                    group = f"{split}:{review_row['review_category']}"
                    _reservoir_add(review_reservoirs, review_seen, group, review_row, examples_per_category, rng)
                if was_truncated:
                    group = f"{split}:truncated_{label_class}"
                    _reservoir_add(truncation_reservoirs, truncation_seen, group, review_row, examples_per_category, rng)
                # Alignment samples are grouped by split and keep exact source text plus decoded tokens.
                _reservoir_add(alignment_reservoirs, alignment_seen, split, review_row, examples_per_category, rng)

        missing_source_rows = len(set(range(len(source))) - row_indices_seen)
        if missing_source_rows:
            alignment_issues.append({"split": split, "row_index": None, "issue": "source_rows_missing_from_tokenized_split", "missing_count": missing_source_rows})
        trunc_table = []
        for (label_class, provenance), total in sorted(total_counts.items()):
            count = truncated_counts[(label_class, provenance)]
            trunc_table.append({"label_class": label_class, "negative_provenance": provenance, "rows": total, "truncated_rows": count, "fraction": count / total if total else 0.0})
        split_reports[split] = {
            "processed_rows": len(source),
            "tokenized_rows": len(ds),
            "unique_pairs": len(pair_first_row),
            "duplicate_pair_rows": sum(1 for item in pair_duplicate_rows if item["split"] == split),
            "duplicate_pair_keys": len({(item["query_id"], item["doc_id"]) for item in pair_duplicate_rows if item["split"] == split}),
            "alignment_issue_count": sum(1 for issue in alignment_issues if issue.get("split") == split),
            "label_issue_count": sum(1 for issue in label_issues if issue.get("split") == split),
            "label_counts": dict(Counter("unjudged" if _is_null(row["label"]) else str(int(row["label"])) for row in ds)),
            "truncation_by_label_and_provenance": trunc_table,
            "truncated_rows": sum(item["truncated_rows"] for item in trunc_table),
            "truncation_fraction": sum(item["truncated_rows"] for item in trunc_table) / len(ds) if len(ds) else 0.0,
            "tokenizer_max_length": max_length,
            "special_tokens_for_pair": special_count,
        }

    print("Stage 3/4: prepare fixed-seed negative and truncation review samples", flush=True)
    # Reservoir sampling caps memory while retaining reproducible examples.
    samples = [row for group in sorted(review_reservoirs) for row in review_reservoirs[group]]
    alignment_samples = [row for split in sorted(alignment_reservoirs) for row in alignment_reservoirs[split]]
    truncation_samples = [row for group in sorted(truncation_reservoirs) for row in truncation_reservoirs[group]]

    print("Stage 4/4: writing audit reports", flush=True)
    issue_columns = ["split", "row_index", "pair_id", "query_id", "doc_id", "issue", "fields", "expected", "source_query_id", "source_doc_id"]
    label_columns = ["split", "row_index", "pair_id", "query_id", "doc_id", "label", "relevance_score", "qrels_score", "issue"]
    pd.DataFrame(alignment_issues).reindex(columns=issue_columns).to_csv(output_dir / "alignment_issues.tsv", sep="\t", index=False)
    pd.DataFrame(label_issues).reindex(columns=label_columns).to_csv(output_dir / "label_issues.tsv", sep="\t", index=False)
    pd.DataFrame(pair_duplicate_rows).to_csv(output_dir / "pair_duplicate_rows.tsv", sep="\t", index=False)
    pd.DataFrame(alignment_samples).to_csv(output_dir / "alignment_decoded_samples.tsv", sep="\t", index=False)
    pd.DataFrame(samples).to_csv(output_dir / "negative_review_samples.tsv", sep="\t", index=False)
    pd.DataFrame(truncation_samples).to_csv(output_dir / "truncation_review_samples.tsv", sep="\t", index=False)
    report = {
        "seed": seed,
        "examples_per_category": examples_per_category,
        "tokenized_dir": str(tokenized_dir.resolve()),
        "processed_dir": str(processed_dir.resolve()),
        "tokenizer": manifest["model"],
        "tokenizer_settings": manifest["tokenizer"],
        "official_qrels": {"name": qrels_name, "revision": qrels_revision, "splits": qrels_splits},
        "official_query_split_overlap_count": len(split_overlap),
        "official_query_split_overlaps": split_overlap[:100],
        "split_reports": split_reports,
        "alignment_issue_count": len(alignment_issues),
        "label_issue_count": len(label_issues),
        "duplicate_pair_row_count": len(pair_duplicate_rows),
        "negative_review_population_counts": dict(review_seen),
        "negative_review_sample_counts": {group: len(rows) for group, rows in review_reservoirs.items()},
        "truncation_review_population_counts": dict(truncation_seen),
        "truncation_review_sample_counts": {group: len(rows) for group, rows in truncation_reservoirs.items()},
        "outputs": {
            "alignment_issues": "alignment_issues.tsv",
            "label_issues": "label_issues.tsv",
            "pair_duplicate_rows": "pair_duplicate_rows.tsv",
            "alignment_decoded_samples": "alignment_decoded_samples.tsv",
            "negative_review_samples": "negative_review_samples.tsv",
            "truncation_review_samples": "truncation_review_samples.tsv",
        },
        "interpretation": {
            "unjudged_candidates": "Unknown relevance; their presence is not evidence of irrelevance.",
            "negative_quality": "Samples support human review; no script can establish irrelevance for unjudged passages.",
            "truncation": "Passage truncation is recomputed with the pinned tokenizer and pair settings; query tokens are preserved.",
        },
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return report
