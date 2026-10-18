"""FiQA validation, negative mining, candidate construction, and export."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tqdm.auto import tqdm

LOGGER = logging.getLogger(__name__)
QRELS_COLUMNS = {"query-id", "corpus-id", "score"}
TEXT_TOKEN = re.compile(r"\$?[a-z0-9]+(?:[./%$-][a-z0-9]+)*", flags=re.IGNORECASE)
OUTPUT_COLUMNS = [
    "split", "query_id", "doc_id", "query_text", "doc_text", "label",
    "relevance_score", "is_judged", "negative_provenance", "candidate_source",
    "candidate_rank", "retrieval_score", "seed",
]


class DataValidationError(ValueError):
    """Raised when an input problem makes the requested split unsafe to use."""


@dataclass
class PreparedBundle:
    corpus: pd.DataFrame
    queries: pd.DataFrame
    qrels: dict[str, pd.DataFrame]
    exclusions: list[dict[str, str]] = field(default_factory=list)
    duplicate_text_groups: dict[str, int] = field(default_factory=dict)


def clean_text(value: Any) -> str:
    """Normalize whitespace only; preserve symbols, punctuation, and numbers."""
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _exclude(log: list[dict[str, str]], entity: str, identifier: Any, reason: str, details: str = "") -> None:
    log.append({"entity": entity, "id": "" if identifier is None else str(identifier), "reason": reason, "details": details})


def _prepare_entity(frame: pd.DataFrame, entity: str, log: list[dict[str, str]]) -> tuple[pd.DataFrame, int]:
    required = {"_id", "title", "text"}
    if not required.issubset(frame.columns):
        raise DataValidationError(f"{entity} must include columns {sorted(required)}; got {list(frame.columns)}")
    result = frame[["_id", "title", "text"]].copy()
    result["_id"] = result["_id"].map(clean_text)
    result["title"] = result["title"].map(clean_text)
    result["text"] = result["text"].map(clean_text)
    result["combined_text"] = (result["title"] + " " + result["text"]).map(clean_text)

    missing_ids = result["_id"].eq("")
    for index in result.index[missing_ids]:
        _exclude(log, entity, None, "missing_id", f"input_row={index}")
    result = result.loc[~missing_ids].copy()

    empty = result["combined_text"].eq("")
    for identifier in result.loc[empty, "_id"]:
        _exclude(log, entity, identifier, "empty_text")
    result = result.loc[~empty].copy()

    conflicting_ids: set[str] = set()
    duplicate_ids = result.loc[result["_id"].duplicated(keep=False), "_id"].unique().tolist()
    for identifier in duplicate_ids:
        group = result.loc[result["_id"].eq(identifier)]
        content = group[["title", "text"]].drop_duplicates()
        if len(content) > 1:
            conflicting_ids.add(identifier)
            for index in group.index:
                _exclude(log, entity, identifier, "conflicting_duplicate_id", f"input_row={index}; all rows with this ID excluded")
        else:
            for index in group.index[1:]:
                _exclude(log, entity, identifier, "duplicate_id_identical", f"input_row={index}; kept first row")
    result = result.loc[~result["_id"].isin(conflicting_ids)].drop_duplicates("_id", keep="first").copy()

    duplicate_text_rows = result["combined_text"].duplicated(keep=False)
    duplicate_groups = int(result.loc[duplicate_text_rows, "combined_text"].nunique())
    for _, group in result.loc[duplicate_text_rows].groupby("combined_text", sort=True):
        ids = ",".join(sorted(group["_id"].astype(str)))
        _exclude(log, entity, ids, "duplicate_text_distinct_ids", f"rows={len(group)}; retained official IDs")
    return result.reset_index(drop=True), duplicate_groups


def prepare_bundle(
    corpus: pd.DataFrame,
    queries: pd.DataFrame,
    qrels_by_split: dict[str, pd.DataFrame],
) -> PreparedBundle:
    """Validate IDs, text, joins, duplicate judgments, and official query splits."""
    exclusions: list[dict[str, str]] = []
    LOGGER.info("Validating corpus rows and IDs (%d input rows)", len(corpus))
    prepared_corpus, corpus_duplicates = _prepare_entity(corpus, "corpus", exclusions)
    LOGGER.info("Corpus validation complete: %d rows retained", len(prepared_corpus))
    LOGGER.info("Validating query rows and IDs (%d input rows)", len(queries))
    prepared_queries, query_duplicates = _prepare_entity(queries, "queries", exclusions)
    LOGGER.info("Query validation complete: %d rows retained", len(prepared_queries))
    corpus_ids = set(prepared_corpus["_id"])
    query_ids = set(prepared_queries["_id"])
    validated_qrels: dict[str, pd.DataFrame] = {}
    split_query_ids: dict[str, set[str]] = {}

    for split, raw in qrels_by_split.items():
        LOGGER.info("Validating %s qrels (%d input judgments)", split, len(raw))
        if not QRELS_COLUMNS.issubset(raw.columns):
            raise DataValidationError(f"{split} qrels must include {sorted(QRELS_COLUMNS)}; got {list(raw.columns)}")
        qrels = raw[["query-id", "corpus-id", "score"]].copy()
        qrels.columns = ["query_id", "doc_id", "score"]
        qrels["query_id"] = qrels["query_id"].map(clean_text)
        qrels["doc_id"] = qrels["doc_id"].map(clean_text)
        qrels["score"] = pd.to_numeric(qrels["score"], errors="coerce")

        invalid = qrels["query_id"].eq("") | qrels["doc_id"].eq("") | qrels["score"].isna()
        for row in qrels.loc[invalid].itertuples(index=False):
            _exclude(exclusions, f"qrels:{split}", f"{row.query_id}/{row.doc_id}", "invalid_judgment", f"score={row.score}")
        qrels = qrels.loc[~invalid].copy()

        for row in qrels.loc[~qrels["query_id"].isin(query_ids)].itertuples(index=False):
            _exclude(exclusions, f"qrels:{split}", f"{row.query_id}/{row.doc_id}", "query_id_not_found")
        for row in qrels.loc[~qrels["doc_id"].isin(corpus_ids)].itertuples(index=False):
            _exclude(exclusions, f"qrels:{split}", f"{row.query_id}/{row.doc_id}", "document_id_not_found")
        qrels = qrels[qrels["query_id"].isin(query_ids) & qrels["doc_id"].isin(corpus_ids)].copy()

        conflicting: set[tuple[str, str]] = set()
        for (query_id, doc_id), group in qrels.groupby(["query_id", "doc_id"], sort=False):
            scores = set(group["score"].astype(float))
            if len(scores) > 1:
                conflicting.add((query_id, doc_id))
                _exclude(exclusions, f"qrels:{split}", f"{query_id}/{doc_id}", "conflicting_judgments", f"scores={sorted(scores)}; pair excluded")
            elif len(group) > 1:
                for _ in range(len(group) - 1):
                    _exclude(exclusions, f"qrels:{split}", f"{query_id}/{doc_id}", "duplicate_judgment_identical", f"score={next(iter(scores))}; kept one")
        if conflicting:
            qrels = qrels.loc[~qrels.set_index(["query_id", "doc_id"]).index.isin(conflicting)]
        qrels = qrels.drop_duplicates(["query_id", "doc_id"], keep="first").reset_index(drop=True)
        validated_qrels[split] = qrels
        split_query_ids[split] = set(qrels["query_id"])
        LOGGER.info("%s qrels validation complete: %d judgments across %d queries", split, len(qrels), len(split_query_ids[split]))

    split_names = sorted(split_query_ids)
    query_text_by_id = prepared_queries.set_index("_id")["combined_text"]
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = split_query_ids[left] & split_query_ids[right]
            if overlap:
                LOGGER.error("Query-ID overlap across official splits %s/%s: %d IDs", left, right, len(overlap))
                raise DataValidationError(f"Query leakage across official splits {left}/{right}: {sorted(overlap)[:10]}")
            left_texts = {query_text_by_id.at[qid] for qid in split_query_ids[left]}
            right_texts = {query_text_by_id.at[qid] for qid in split_query_ids[right]}
            duplicate_queries = left_texts & right_texts
            if duplicate_queries:
                LOGGER.error("Exact query-text overlap across official splits %s/%s: %d texts", left, right, len(duplicate_queries))
                raise DataValidationError(f"Duplicate query text across official splits {left}/{right}: {len(duplicate_queries)} text(s)")

    return PreparedBundle(
        corpus=prepared_corpus,
        queries=prepared_queries,
        qrels=validated_qrels,
        exclusions=exclusions,
        duplicate_text_groups={"corpus": corpus_duplicates, "queries": query_duplicates},
    )


def tokenize_for_retrieval(text: str) -> list[str]:
    """Tokenize for lexical retrieval without altering the exported source text."""
    return TEXT_TOKEN.findall(clean_text(text).lower())


class BM25Index:
    """Compact sparse BM25 index with deterministic score and tie ordering."""

    def __init__(self, corpus: pd.DataFrame, *, k1: float = 1.2, b: float = 0.75) -> None:
        if corpus.empty:
            raise DataValidationError("Cannot build a retrieval index from an empty corpus")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and b between 0 and 1")
        self.corpus = corpus.reset_index(drop=True)
        self.k1 = float(k1)
        self.b = float(b)
        self.vectorizer = CountVectorizer(
            tokenizer=tokenize_for_retrieval,
            token_pattern=None,
            lowercase=False,
            dtype=np.float32,
        )
        counts = self.vectorizer.fit_transform(self.corpus["combined_text"].tolist())
        self.matrix = counts.tocsc()
        del counts
        self.doc_lengths = np.asarray(self.matrix.sum(axis=1)).ravel().astype(np.float32)
        self.average_length = max(float(self.doc_lengths.mean()), 1.0)
        self.doc_freq = np.diff(self.matrix.indptr).astype(np.float32)
        n_docs = len(self.corpus)
        self.idf = np.log1p((n_docs - self.doc_freq + 0.5) / (self.doc_freq + 0.5)).astype(np.float32)
        self.ids = self.corpus["_id"].astype(str).to_numpy()

    def search(self, query_text: str, top_k: int, *, partial_top_k: bool = False) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        tokens = sorted(set(tokenize_for_retrieval(query_text)))
        scores = np.zeros(len(self.ids), dtype=np.float32)
        vocabulary = self.vectorizer.vocabulary_
        for token in tokens:
            term_index = vocabulary.get(token)
            if term_index is None:
                continue
            start, end = self.matrix.indptr[term_index : term_index + 2]
            rows = self.matrix.indices[start:end]
            term_frequency = self.matrix.data[start:end]
            denominator = term_frequency + self.k1 * (
                1.0 - self.b + self.b * self.doc_lengths[rows] / self.average_length
            )
            scores[rows] += self.idf[term_index] * term_frequency * (self.k1 + 1.0) / denominator
        hits = np.flatnonzero(scores > 0)
        if not len(hits):
            return []
        if partial_top_k and len(hits) > top_k:
            # Select the score cutoff in linear time, then resolve cutoff ties
            # by ID exactly as the standard full sort does.
            hit_scores = scores[hits]
            cutoff_index = len(hits) - top_k
            cutoff_score = np.partition(hit_scores, cutoff_index)[cutoff_index]
            better = hits[hit_scores > cutoff_score]
            tied = hits[hit_scores == cutoff_score]
            tied_needed = top_k - len(better)
            tied_by_id = tied[np.argsort(self.ids[tied], kind="stable")[:tied_needed]]
            rows = np.concatenate((better, tied_by_id))
        else:
            rows = hits
        order = np.lexsort((self.ids[rows], -scores[rows]))[:top_k]
        return [
            {"doc_id": str(self.ids[row]), "retrieval_score": float(scores[row]), "candidate_rank": rank}
            for rank, row in enumerate(rows[order], start=1)
        ]

    def search_queries(
        self,
        queries: pd.DataFrame,
        top_k: int,
        *,
        partial_top_k: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        rows = queries.sort_values("_id", kind="stable").to_dict("records")
        results: dict[str, list[dict[str, Any]]] = {}
        for row in tqdm(rows, desc="BM25 query retrieval", unit="query", dynamic_ncols=True):
            query_id = str(row["_id"])
            results[query_id] = self.search(row["combined_text"], top_k, partial_top_k=partial_top_k)
        return results


def _stable_rng(seed: int, *parts: str) -> np.random.Generator:
    payload = "|".join([str(seed), *parts]).encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return np.random.default_rng(derived_seed)


def _sample_eligible_without_replacement(
    rng: np.random.Generator,
    corpus_ids: np.ndarray,
    excluded: set[str],
    count: int,
) -> list[str]:
    """Uniformly sample eligible IDs without materializing/sorting the pool."""
    eligible_count = len(corpus_ids) - len(excluded)
    count = min(count, eligible_count)
    if count <= 0:
        return []
    if count == eligible_count:
        # Rare boundary case; materialize once instead of risking a long
        # rejection loop when nearly every corpus row is requested.
        eligible = [str(doc_id) for doc_id in corpus_ids if str(doc_id) not in excluded]
        return [str(doc_id) for doc_id in rng.choice(eligible, size=count, replace=False)]

    chosen: list[str] = []
    chosen_set: set[str] = set()
    attempts = 0
    while len(chosen) < count:
        remaining = count - len(chosen)
        batch_size = max(16, remaining * 3)
        proposals = rng.choice(corpus_ids, size=batch_size, replace=True)
        for proposal in proposals:
            doc_id = str(proposal)
            if doc_id not in excluded and doc_id not in chosen_set:
                chosen.append(doc_id)
                chosen_set.add(doc_id)
                if len(chosen) == count:
                    break
        attempts += batch_size
        if attempts > len(corpus_ids) * 10 and len(chosen) < count:
            eligible = [str(doc_id) for doc_id in corpus_ids if str(doc_id) not in excluded and str(doc_id) not in chosen_set]
            chosen.extend(str(doc_id) for doc_id in rng.choice(eligible, size=count - len(chosen), replace=False))
    return chosen


def _base_row(
    split: str,
    query_id: str,
    doc_id: str,
    query_text: str,
    doc_text: str,
    seed: int,
    *,
    label: int | None,
    relevance_score: float | None,
    is_judged: bool,
    negative_provenance: str,
    candidate_source: str,
    candidate_rank: int | None = None,
    retrieval_score: float | None = None,
) -> dict[str, Any]:
    return {
        "split": split,
        "query_id": query_id,
        "doc_id": doc_id,
        "query_text": query_text,
        "doc_text": doc_text,
        "label": label,
        "relevance_score": relevance_score,
        "is_judged": int(is_judged),
        "negative_provenance": negative_provenance,
        "candidate_source": candidate_source,
        "candidate_rank": candidate_rank,
        "retrieval_score": retrieval_score,
        "seed": seed,
    }


def _build_training_examples(
    bundle: PreparedBundle,
    retrieved: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
    negatives_per_positive: int = 4,
    hard_negative_fraction: float = 0.5,
    reuse_pool_per_query: bool,
    rejection_sampling: bool = False,
    include_text: bool = True,
    sort_output: bool = True,
) -> pd.DataFrame:
    """Build training pairs with explicit provenance and known-positive exclusion."""
    if negatives_per_positive < 0:
        raise ValueError("negatives_per_positive cannot be negative")
    if not 0 <= hard_negative_fraction <= 1:
        raise ValueError("hard_negative_fraction must be between 0 and 1")
    if "train" not in bundle.qrels:
        raise DataValidationError("No official train qrels were provided")

    corpus_by_id = bundle.corpus.set_index("_id")
    query_by_id = bundle.queries.set_index("_id")
    qrels_all = pd.concat(bundle.qrels.values(), ignore_index=True)
    known_positive_by_query = {
        str(qid): set(group.loc[group["score"] > 0, "doc_id"].astype(str))
        for qid, group in qrels_all.groupby("query_id", sort=False)
    }
    judged_score = {
        (str(row.query_id), str(row.doc_id)): row.score
        for row in qrels_all.itertuples(index=False)
    }
    train = bundle.qrels["train"]
    positives = train[train["score"] > 0].sort_values(["query_id", "doc_id"], kind="stable")
    hard_count = round(negatives_per_positive * hard_negative_fraction)
    easy_count = negatives_per_positive - hard_count
    records: list[dict[str, Any]] = []
    cached_query_id: str | None = None
    cached_hard_selected: list[dict[str, Any]] = []
    cached_available = np.array([], dtype=object)
    cached_excluded: set[str] = set()
    corpus_ids = corpus_by_id.index.astype(str).to_numpy()

    for positive in tqdm(positives.itertuples(index=False), total=len(positives), desc="Training pairs and negatives", unit="positive", dynamic_ncols=True):
        qid, positive_id = str(positive.query_id), str(positive.doc_id)
        query_text = query_by_id.at[qid, "combined_text"] if include_text else ""
        positive_text = corpus_by_id.at[positive_id, "combined_text"] if include_text else ""
        records.append(
            _base_row(
                "train", qid, positive_id, query_text, positive_text, seed,
                label=1, relevance_score=positive.score, is_judged=True,
                negative_provenance="", candidate_source="qrels_positive",
            )
        )

        positives_for_query = known_positive_by_query.get(qid, set())
        if not reuse_pool_per_query or qid != cached_query_id:
            hard_pool: list[dict[str, Any]] = []
            seen: set[str] = set(positives_for_query)
            for candidate in retrieved.get(qid, []):
                doc_id = str(candidate["doc_id"])
                if doc_id in seen or doc_id not in corpus_by_id.index:
                    continue
                seen.add(doc_id)
                hard_pool.append(candidate)
            cached_hard_selected = hard_pool[:hard_count]
            cached_excluded = positives_for_query | {str(row["doc_id"]) for row in hard_pool}
            if rejection_sampling:
                cached_available = np.array([], dtype=object)
            else:
                # The standard mode retains its original lexicographic pool and
                # therefore exactly the same samples as before this fast path.
                cached_available = np.array(sorted(set(corpus_ids) - cached_excluded), dtype=object)
            cached_query_id = qid
        hard_selected = cached_hard_selected
        for candidate in hard_selected:
            doc_id = str(candidate["doc_id"])
            score = judged_score.get((qid, doc_id))
            provenance = "retrieved_judged_nonrelevant" if score is not None and score <= 0 else "retrieved_unjudged_false_negative_risk"
            records.append(
                _base_row(
                    "train", qid, doc_id, query_text,
                    corpus_by_id.at[doc_id, "combined_text"] if include_text else "", seed,
                    label=0, relevance_score=score, is_judged=score is not None,
                    negative_provenance=provenance, candidate_source="bm25_hard_negative",
                    candidate_rank=candidate.get("candidate_rank"), retrieval_score=candidate.get("retrieval_score"),
                )
            )

        remaining = negatives_per_positive - len(hard_selected)
        requested_easy = max(easy_count, remaining)
        n_random = min(requested_easy, len(corpus_ids) - len(cached_excluded)) if rejection_sampling else min(requested_easy, len(cached_available))
        if n_random:
            rng = _stable_rng(seed, qid, positive_id, "random")
            chosen = (
                _sample_eligible_without_replacement(rng, corpus_ids, cached_excluded, n_random)
                if rejection_sampling
                else rng.choice(cached_available, size=n_random, replace=False).tolist()
            )
            fallback_remaining = max(0, hard_count - len(hard_selected))
            for doc_id in chosen:
                score = judged_score.get((qid, str(doc_id)))
                if fallback_remaining:
                    provenance = "random_fallback_hard_pool_shortage"
                    source = "random_fallback"
                    fallback_remaining -= 1
                else:
                    provenance = "random_judged_nonrelevant" if score is not None and score <= 0 else "random_unjudged_false_negative_risk"
                    source = "random_easy_negative"
                records.append(
                    _base_row(
                        "train", qid, str(doc_id), query_text,
                        corpus_by_id.at[str(doc_id), "combined_text"] if include_text else "", seed,
                        label=0, relevance_score=score, is_judged=score is not None,
                        negative_provenance=provenance, candidate_source=source,
                    )
                )
    result = pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)
    if sort_output and not result.empty:
        result = result.sort_values(["query_id", "doc_id", "label"], ascending=[True, True, False], kind="stable").reset_index(drop=True)
    return result


def build_training_examples(
    bundle: PreparedBundle,
    retrieved: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
    negatives_per_positive: int = 4,
    hard_negative_fraction: float = 0.5,
) -> pd.DataFrame:
    """Build training examples using the standard per-positive pool path."""
    return _build_training_examples(
        bundle, retrieved, seed=seed,
        negatives_per_positive=negatives_per_positive,
        hard_negative_fraction=hard_negative_fraction,
        reuse_pool_per_query=False,
    )


def build_training_examples_fast(
    bundle: PreparedBundle,
    retrieved: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
    negatives_per_positive: int = 4,
    hard_negative_fraction: float = 0.5,
    include_text: bool = True,
) -> pd.DataFrame:
    """Build reproducible examples with query-level pools and rejection sampling."""
    return _build_training_examples(
        bundle, retrieved, seed=seed,
        negatives_per_positive=negatives_per_positive,
        hard_negative_fraction=hard_negative_fraction,
        reuse_pool_per_query=True,
        rejection_sampling=True,
        include_text=include_text,
        sort_output=False,
    )


def build_evaluation_candidates(
    bundle: PreparedBundle,
    split: str,
    retrieved: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
    include_text: bool = True,
) -> pd.DataFrame:
    """Export retrieved candidates as-is; unjudged labels remain null."""
    if split not in bundle.qrels:
        raise DataValidationError(f"Missing qrels for split {split}")
    query_by_id = bundle.queries.set_index("_id")
    corpus_by_id = bundle.corpus.set_index("_id")
    qrels = bundle.qrels[split]
    judgment = {(str(row.query_id), str(row.doc_id)): row.score for row in qrels.itertuples(index=False)}
    records: list[dict[str, Any]] = []
    query_ids = sorted(set(qrels["query_id"]))
    for qid in tqdm(query_ids, desc=f"{split.title()} candidate rows", unit="query", dynamic_ncols=True):
        if qid not in query_by_id.index:
            continue
        query_text = query_by_id.at[qid, "combined_text"] if include_text else ""
        for candidate in retrieved.get(qid, []):
            doc_id = str(candidate["doc_id"])
            if doc_id not in corpus_by_id.index:
                continue
            score = judgment.get((qid, doc_id))
            judged = score is not None
            records.append(
                _base_row(
                    split, qid, doc_id, query_text,
                    corpus_by_id.at[doc_id, "combined_text"] if include_text else "", seed,
                    label=int(score > 0) if judged else None,
                    relevance_score=score,
                    is_judged=judged,
                    negative_provenance="",
                    candidate_source="bm25_evaluation_candidate",
                    candidate_rank=candidate.get("candidate_rank"),
                    retrieval_score=candidate.get("retrieval_score"),
                )
            )
    result = pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)
    if include_text and not result.empty:
        result = result.sort_values(["query_id", "candidate_rank"], kind="stable").reset_index(drop=True)
    return result


def _dataset_to_frame(dataset: Any) -> pd.DataFrame:
    return dataset.to_pandas()


def load_huggingface_bundle(config: dict[str, Any]) -> tuple[PreparedBundle, dict[str, str]]:
    """Load pinned FiQA corpus, query, and official qrels snapshots from HF."""
    from datasets import load_dataset

    data_cfg = config["dataset"]
    LOGGER.info("Loading corpus snapshot %s@%s", data_cfg["name"], data_cfg["revision"])
    corpus = load_dataset(data_cfg["name"], "corpus", split="corpus", revision=data_cfg["revision"])
    LOGGER.info("Corpus loaded: %d rows", len(corpus))
    LOGGER.info("Loading queries snapshot %s@%s", data_cfg["name"], data_cfg["revision"])
    queries = load_dataset(data_cfg["name"], "queries", split="queries", revision=data_cfg["revision"])
    LOGGER.info("Queries loaded: %d rows", len(queries))
    qrels_cfg = data_cfg["qrels"]
    split_names = qrels_cfg.get("split_names", {"train": "train", "validation": "validation", "test": "test"})
    raw_qrels = {}
    for canonical, hf_split in split_names.items():
        LOGGER.info("Loading official %s qrels snapshot %s@%s (HF split=%s)", canonical, qrels_cfg["name"], qrels_cfg["revision"], hf_split)
        raw_qrels[canonical] = load_dataset(qrels_cfg["name"], split=hf_split, revision=qrels_cfg["revision"])
        LOGGER.info("%s qrels loaded: %d judgments", canonical, len(raw_qrels[canonical]))
    fingerprints = {
        "corpus": getattr(corpus, "_fingerprint", "unknown"),
        "queries": getattr(queries, "_fingerprint", "unknown"),
        **{f"qrels_{split}": getattr(dataset, "_fingerprint", "unknown") for split, dataset in raw_qrels.items()},
    }
    LOGGER.info("Converting loaded datasets to pandas frames and validating joins/splits")
    bundle = prepare_bundle(_dataset_to_frame(corpus), _dataset_to_frame(queries), {k: _dataset_to_frame(v) for k, v in raw_qrels.items()})
    LOGGER.info("Dataset validation complete: corpus=%d, queries=%d, exclusions=%d", len(bundle.corpus), len(bundle.queries), len(bundle.exclusions))
    return bundle, fingerprints


def write_exclusions(exclusions: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(exclusions, columns=["entity", "id", "reason", "details"]).to_csv(path, sep="\t", index=False)


def _write_pair_tsv_in_chunks(
    frame: pd.DataFrame,
    path: Path,
    bundle: PreparedBundle,
    *,
    chunk_size: int = 2048,
) -> None:
    """Restore text by ID and stream a compact pair frame to the TSV."""
    query_text = bundle.queries.set_index("_id")["combined_text"].to_dict()
    document_text = bundle.corpus.set_index("_id")["combined_text"].to_dict()
    if frame.empty:
        frame.to_csv(path, sep="\t", index=False, na_rep="")
        return
    starts = range(0, len(frame), chunk_size)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for start in tqdm(starts, desc=f"Writing {path.name}", unit="chunk", dynamic_ncols=True):
            chunk = frame.iloc[start : start + chunk_size].copy()
            chunk["query_text"] = chunk["query_id"].map(query_text)
            chunk["doc_text"] = chunk["doc_id"].map(document_text)
            chunk.to_csv(handle, sep="\t", index=False, header=(start == 0), na_rep="")


def export_pipeline(
    bundle: PreparedBundle,
    *,
    out_dir: Path,
    seed: int,
    retrieval_depth: int,
    eval_top_k: int,
    bm25_k1: float,
    bm25_b: float,
    negatives_per_positive: int,
    hard_negative_fraction: float,
    dataset_revisions: dict[str, str],
    fingerprints: dict[str, str],
    config: dict[str, Any],
    fast_mode: bool = False,
) -> dict[str, Any]:
    """Retrieve candidates, export train/evaluation TSVs, and write a manifest."""
    if retrieval_depth < 1 or eval_top_k < 1:
        raise ValueError("retrieval_depth and eval_top_k must be at least 1")
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Stage 1/6: building sparse BM25 index for %d corpus documents", len(bundle.corpus))
    index = BM25Index(bundle.corpus, k1=bm25_k1, b=bm25_b)
    LOGGER.info("BM25 index ready")
    all_qrels = pd.concat(bundle.qrels.values(), ignore_index=True)
    query_ids = sorted(set(all_qrels["query_id"]))
    qmap = bundle.queries.set_index("_id")
    retrieval_depth = max(retrieval_depth, eval_top_k)
    LOGGER.info("Stage 2/6: retrieving up to %d BM25 candidates for %d qrels queries", retrieval_depth, len(query_ids))
    retrieved = index.search_queries(
        qmap.loc[query_ids].reset_index(), retrieval_depth, partial_top_k=fast_mode,
    )
    retrieved_eval = {qid: candidates[:eval_top_k] for qid, candidates in retrieved.items()}

    LOGGER.info("Stage 3/6: creating training positives and sampled negatives")
    if fast_mode:
        train_rows = build_training_examples_fast(
            bundle, retrieved, seed=seed,
            negatives_per_positive=negatives_per_positive,
            hard_negative_fraction=hard_negative_fraction,
            include_text=False,
        )
    else:
        train_rows = build_training_examples(
            bundle, retrieved, seed=seed,
            negatives_per_positive=negatives_per_positive,
            hard_negative_fraction=hard_negative_fraction,
        )
    validation_split = "validation" if "validation" in bundle.qrels else "dev"
    LOGGER.info("Stage 4/6: creating unbalanced %s evaluation candidates", validation_split)
    validation_rows = build_evaluation_candidates(
        bundle, validation_split, retrieved_eval, seed=seed, include_text=not fast_mode,
    )
    LOGGER.info("Stage 5/6: creating unbalanced test evaluation candidates")
    test_rows = build_evaluation_candidates(
        bundle, "test", retrieved_eval, seed=seed, include_text=not fast_mode,
    )
    outputs = {
        "train": train_rows,
        "validation": validation_rows,
        "test": test_rows,
    }
    output_paths: dict[str, str] = {}
    output_hashes: dict[str, str] = {}
    output_paths["exclusions"] = "exclusions.tsv"
    LOGGER.info("Stage 6/6: writing data artifacts to %s", out_dir.resolve())
    files_to_write = list(outputs.items()) + [("exclusions", pd.DataFrame(bundle.exclusions, columns=["entity", "id", "reason", "details"]))]
    for name, frame in tqdm(files_to_write, desc="Writing TSV files", unit="file", dynamic_ncols=True):
        path = out_dir / f"{name}.tsv"
        if fast_mode and name in outputs:
            _write_pair_tsv_in_chunks(frame, path, bundle)
        else:
            frame.to_csv(path, sep="\t", index=False, na_rep="")
        if name != "exclusions":
            output_paths[name] = path.name
        output_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    train_negatives = train_rows[train_rows["label"] == 0]
    manifest: dict[str, Any] = {
        "dataset": dataset_revisions,
        "dataset_fingerprints": fingerprints,
        "seed": seed,
        "configuration": config,
        "software": {
            "python": sys.version.split()[0],
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
            "scikit-learn": importlib.metadata.version("scikit-learn"),
            "datasets": importlib.metadata.version("datasets"),
        },
        "split_policy": {"train": "official train", "validation": "official dev/validation", "test": "official test"},
        "cleaning": "collapse leading/trailing and repeated whitespace only; retain symbols and numeric text",
        "negative_policy": {
            "positive_definition": "qrels score > 0",
            "ratio": f"1:{negatives_per_positive} positive:negative target per positive qrel",
            "hard_negative_fraction": hard_negative_fraction,
            "hard_source": "BM25 retrieved candidates not known positive for the query",
            "easy_source": "uniform random corpus documents not known positive for the query",
            "false_negative_risk": "retrieved or random unjudged documents may be relevant; provenance and is_judged are retained",
            "evaluation": "retrieved candidate lists are not balanced; unjudged candidate labels are blank",
        },
        "pipeline_variant": {
            "name": "fast" if fast_mode else "standard",
            "random_negative_sampling": "seeded rejection sampling from eligible corpus IDs" if fast_mode else "seeded sampling from sorted eligible corpus IDs",
            "query_pool_reuse": fast_mode,
            "retrieval_top_k": "partial score-cutoff selection with exact document-ID tie resolution" if fast_mode else "full deterministic sort",
            "row_sorting": "stable generation order; DataFrame sort skipped" if fast_mode else "final DataFrame sort by query/document ID",
            "text_export": "restored by IDs and written in chunks" if fast_mode else "written from full in-memory frames",
        },
        "counts": {
            "corpus_documents": len(bundle.corpus),
            "queries": len(bundle.queries),
            "qrels": {split: {"judgments": len(frame), "queries": int(frame.query_id.nunique()), "positives": int((frame.score > 0).sum())} for split, frame in bundle.qrels.items()},
            "examples": {name: len(frame) for name, frame in outputs.items()},
            "train_negatives_by_provenance": {str(k): int(v) for k, v in train_negatives.negative_provenance.value_counts().items()},
            "exclusions_by_reason": {str(k): int(v) for k, v in pd.Series([e["reason"] for e in bundle.exclusions]).value_counts().items()},
            "duplicate_text_groups": bundle.duplicate_text_groups,
        },
        "artifacts": {"files": output_paths, "sha256": output_hashes},
        "limitations": [
            "Unjudged pairs are not verified negatives; sampled labels have documented false-negative risk.",
            "Evaluation candidates are BM25 top-k; qrels positives outside that list are not injected.",
            "An unjudged evaluation candidate has a blank label and must remain masked in custom metrics.",
        ],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("Wrote manifest: %s", manifest_path.resolve())
    return manifest
