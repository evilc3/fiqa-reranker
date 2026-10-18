# FiQA Cross-Encoder Reranker

Learning project for a reproducible BEIR/FiQA reranker pipeline. The intended
workflow is to inspect the dataset and hardware, validate official splits and
judgments, select a baseline after checking training overlap, then train and
evaluate without using test results for tuning.

## Git LFS setup

Large datasets and model weight files in this repository are stored with Git
Large File Storage (LFS). Install Git LFS once, then clone and fetch the LFS
objects:

```bash
# Install Git LFS using your operating system's package manager, or see
# https://git-lfs.com/ for the official installer.
git lfs install
git clone <repository-url>
cd fiqa-reranker
git lfs pull
```

If you already cloned the repository, run `git lfs install` and `git lfs pull`
from the repository directory. Without LFS, Git checks out small pointer files
instead of the dataset and model contents. Pushing these artifacts also uses
your Git host's LFS storage and bandwidth quota.

## Experiment report: 18 October 2026

The `easy-batch32-v1` experiment improved FiQA ranking over the untouched
pretrained model on both validation and test. Epoch 2 was selected using
validation nDCG@10 and restored before the final test evaluation. Results below
come from saved JSON artifacts, cross-checked against the completed run logs.

### What changed

The earlier fast pipeline used positives from official qrels (`score > 0`) and
four negatives per positive: two random corpus passages and two BM25 passages.
Its training data contained repeated query/passage pairs, and unjudged retrieved
passages could be valid answers despite receiving training label zero.

We created a separate easy-only dataset with the same 1:4 ratio and number of
training examples. It samples random passages from sorted corpus IDs using a
SHA256-derived per-query RNG, excludes all known positives for that query, and
samples unique negative pairs without replacement. Seed: **42**. There are
**14,131 positives and 56,524 random negatives**, with zero duplicate training
query/passage pairs. Random sampling still cannot guarantee irrelevance or a
different topic; that false-negative risk is recorded in the manifest.

Official query splits were preserved. Easy validation/test TSVs are byte-identical
copies of the original fast pipeline's BM25 evaluation candidates: no balancing
or positive injection. Unjudged evaluation labels remain blank; these rows are
scored and ranked, and do not contribute to training loss. Metrics use original
official qrels, including positive judgments outside the candidate list.

### Datasets and locations

All paths below are relative to the project root:
`C:\Users\Clive\Desktop\fiqa-reranker\`.

| Dataset artifact | Saved directory | Train rows | Validation rows | Test rows |
|---|---|---:|---:|---:|
| Original mixed random/BM25 pairs and candidates | `data/processed/fiqa_fast/` | 70,655 | 50,000 | 64,754 |
| Tokenized original mixed dataset | `data/tokenized/fiqa_fast_minilm_l6_v2/` | 70,655 | 50,000 | 64,754 |
| Easy-only pairs and unchanged evaluation candidates | `data/processed/fiqa_easy/` | 70,655 | 50,000 | 64,754 |
| Tokenized easy dataset used for this run | `data/tokenized/fiqa_easy_minilm_l6_v2/` | 70,655 | 50,000 | 64,754 |

Processed directories contain `train.tsv`, `validation.tsv`, `test.tsv`,
`exclusions.tsv`, and `manifest.json`. Tokenized directories contain separate
`train/`, `validation/`, and `test/` Arrow datasets, `manifest.json`, and
`exclusions.jsonl`. Manifests record configuration, provenance, seeds, counts,
and checksums. Validation has 500 queries; test has 648 queries.

Raw source snapshots are `BeIR/fiqa` at
`979c07a7cb5ccc6ca009792241fa1250b98055dd` and `BeIR/fiqa-qrels` at
`252958f2d646e22cab6d0c72dd3f0d5de6d0655a`. Hugging Face manages the raw download
cache; the paths above are the exported experiment datasets.

Tokenization jointly encodes query and passage with a **256-token combined
limit**, including special tokens. Only passages are truncated; overlong queries
are rejected. The easy export rejected zero queries. Passage truncation affected
23.76% of training rows, 38.06% of validation rows, and 38.07% of test rows.
Arrow sequences are saved without padding; training dynamically pads each batch.

### Training settings and results

Starting model: `cross-encoder/ms-marco-MiniLM-L6-v2`, revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`. Configuration:
[step3_easy.yaml](configs/step3_easy.yaml). The run used a GTX 1650 GPU, batch
size 32, two epochs, explicit BCEWithLogitsLoss, AdamW with LR `2e-5` and weight
decay `0.01`, 10% linear warmup followed by linear decay, gradient clipping
`1.0`, mixed precision, and early-stopping patience 2. Validation ran each epoch.

| Checkpoint | Train BCE | Validation nDCG@10 | Validation MRR@10 |
|---|---:|---:|---:|
| Untouched pretrained baseline | N/A | 0.328590 | 0.407556 |
| Easy epoch 1 | 0.158714 | 0.348972 | 0.431355 |
| Easy epoch 2 (selected) | 0.091384 | **0.351027** | **0.430853** |

Epoch 2 had the higher validation nDCG@10, which is the configured selection
metric. Its MRR@10 was slightly below epoch 1. `stale=0/2` means nDCG improved,
so the early-stopping counter reset; training finished at the two-epoch limit.

The untouched pretrained test baseline was measured separately using the same
tokenized test candidates and official qrels. Improvements use full-precision
saved values; displayed scores are rounded to six decimals.

| Split / metric | Pretrained baseline | Selected easy model | Absolute gain | Relative improvement |
|---|---:|---:|---:|---:|
| Validation nDCG@10 | 0.328590 | 0.351027 | +0.022437 | **+6.83%** |
| Validation MRR@10 | 0.407556 | 0.430853 | +0.023297 | **+5.72%** |
| Test nDCG@10 | 0.320595 | 0.336985 | +0.016390 | **+5.11%** |
| Test MRR@10 | 0.399135 | 0.412454 | +0.013319 | **+3.34%** |

Relative improvement is `(fine_tuned - baseline) / baseline * 100`.
Validation and test use different queries, so compare each result with its own
split's baseline. The lower test score than validation does not indicate a
checkpoint reload problem.

For comparison, the earlier mixed-negative run selected epoch 2 with validation
nDCG@10 **0.156696**, MRR@10 **0.174607**, and test nDCG@10 **0.140719**,
MRR@10 **0.164272**. Easy-only training recovered from that regression and
exceeded the original pretrained baseline. This supports investigating the
original negative construction, but does not prove BM25 alone caused the drop:
the easy experiment changed both negative source and repeated-pair weighting.
This is one seed/run, with no statistical significance estimate. Continue to
make training/model-selection decisions using validation, rather than test.

### Final model and experiment artifacts

**Final selected model:**
`C:\Users\Clive\Desktop\fiqa-reranker\outputs\step3_easy\easy-batch32-v1\best_model\`

This is the epoch-2 model with `model.safetensors`, `config.json`,
`tokenizer.json`, and `tokenizer_config.json`. The untouched pretrained snapshot
is stored separately at `checkpoints/ms-marco-MiniLM-L6-v2/`.

The run directory `outputs/step3_easy/easy-batch32-v1/` also contains per-epoch
checkpoints, validation score TSVs, `training_history.json`, `test_scores.tsv`,
`test_metrics.json`, `run_metadata.json`, and `resolved_config.yaml`.
The validation baseline is saved under its `pretraining_baseline/` subdirectory.
The separate pretrained test baseline, including `test_scores.tsv`,
`metrics.json`, and `resolved_config.yaml`, is saved at:
`reports/baseline_easy/pretrained_test_20261018_125439_864939_utc/`.
Earlier mixed-negative training results are stored under `outputs/step3/`.

Reproduction commands, from the project root (use a new run name for another
training run):

```bash
uv run python scripts/prepare_fiqa_data_easy.py
uv run python scripts/tokenize_fiqa_pairs.py --input-dir data/processed/fiqa_easy --output-dir data/tokenized/fiqa_easy_minilm_l6_v2 --max-length 256
uv run python scripts/train_reranker.py --config configs/step3_easy.yaml --training-run-name easy-batch32-v1
uv run python scripts/score_baseline.py --config configs/step3_easy.yaml --split test
```

Easy preprocessing requires the original `data/processed/fiqa_fast/` evaluation
source and refuses to overwrite a nonempty easy output directory. Existing
training run names are also protected. See the preprocessing sections below to
recreate source candidates or choose fresh output directories.

## Setup

Requires Python 3.12 and `uv`. The lockfile pins the resolved dependencies;
PyTorch is pinned to the official CUDA 12.6 wheel, tested on a GTX 1650 with
4 GB VRAM. Recreate the environment with:

```powershell
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Set experiment values in `configs/base.yaml`. Before downloading data or
selecting a model, record immutable revisions and inspect their licenses,
provenance, and training overlap. The baseline and training steps will preserve
the official BEIR splits. Unjudged passages must not be treated as known
irrelevant examples.

## Structure
- `src/fiqa_reranker/`: reusable pipeline code
- `configs/`: experiment configuration
- `scripts/`: runnable entry points
- `tests/`: validation and smoke tests
- `notebooks/`: exploration only
- `reports/`: experiment summaries

Dataset files, checkpoints, and generated outputs are excluded from Git. Keep
experiment configurations and metrics with each run; do not tune on test
metrics. External publication is out of scope unless explicitly requested.

## Step 2: prepare training and evaluation data

The Python pipeline reads the pinned `BeIR/fiqa` corpus/query snapshots and
`BeIR/fiqa-qrels` judgments in [data.yaml](configs/data.yaml). It validates
IDs and joins, cleans text by trimming and collapsing whitespace only, and
retains the official train/dev/test query splits. It rejects query-ID or exact
query-text overlap across those splits. The corpus is shared across retrieval
splits by BEIR design; document IDs are not reassigned into new splits. The
qrels repository exposes its `dev.tsv` through the `validation` split name in
Hugging Face Datasets; the export is named `validation.tsv` and the manifest
records that mapping.

From the project root, run:

```powershell
uv run python scripts/prepare_fiqa_data.py
```

An alternative entry point, `uv run python scripts/prepare_fiqa_data_fast.py`,
reuses each query's eligible negative set, samples easy negatives with seeded
rejection sampling, uses partial top-k retrieval, skips the final DataFrame
sort, and writes text-rich TSVs in chunks after looking up IDs. Its random
random negative samples may differ from the standard script, while retrieval
candidate ordering preserves the standard score/ID tie break. Repeat fast runs
with the same pinned inputs and seed are reproducible. The manifest records the
fast-mode policies. It writes by default to `data/processed/fiqa_fast/`, so
it will not overwrite the standard output directory. You can provide
`--output-dir PATH` to either command.

The script logs named stages for dataset loading and validation, index building,
retrieval, pair/candidate creation, and export. `tqdm` bars show per-query BM25
retrieval, per-positive training pair creation, per-query evaluation candidate
creation, and TSV writes. At completion it prints the absolute output paths,
row counts, exclusion totals, and the first five rows of each TSV. Text fields
are shortened in previews so the terminal output stays readable. Use
`--preview-rows 0` to skip previews or `--preview-rows N` to change their size.

To use another output directory or config, pass `--output-dir PATH` and/or
`--config PATH`. The command builds a sparse BM25 index, creates training pairs,
and writes these files under `data/processed/fiqa/` (ignored by Git):

- `train.tsv`: every official train qrel with score greater than zero is a
  positive. The default target is four negatives per positive, half drawn from
  BM25's highest-ranked eligible candidates and half sampled uniformly from
  the corpus. Sampling is configurable in `configs/data.yaml`.
- `validation.tsv` and `test.tsv`: BM25 top-k candidate lists, not balanced
  samples. The pipeline does not append positives that BM25 failed to retrieve.
  Judged candidates receive labels from qrels; unjudged candidates have blank
  labels and `is_judged=0`.
- `exclusions.tsv`: rows excluded for invalid IDs/text/joins, conflicting
  judgments, or other validation findings. Duplicate text under distinct IDs
  is reported and retained because the official IDs are preserved.
- `manifest.json`: source revisions and fingerprints, config, seed, cleaning
  and negative policies, counts, exclusion summaries, and SHA-256 hashes.

All pair rows include query/document IDs and text, label, relevance score,
judgment status, negative provenance, candidate source/rank, retrieval score,
and seed. Train negatives exclude every known positive for that query.
Retrieved and random unjudged passages can still be relevant: the manifest and
`negative_provenance` flag this false-negative risk. Do not interpret blank
evaluation labels as zero or use test results for tuning. Candidate coverage is
limited by BM25 recall; report that limitation with reranker metrics.

Run the data pipeline unit tests with:

```powershell
uv run pytest -q
```

## Step 2.2: download the baseline model and tokenize pairs

`configs/base.yaml` pins `cross-encoder/ms-marco-MiniLM-L6-v2` to an immutable
Hub commit. The command below downloads the model and matching tokenizer into
the Git-ignored `checkpoints/` directory, verifies that both reload offline,
and tokenizes the processed train, validation, and test TSVs:

```powershell
uv run python scripts/tokenize_fiqa_pairs.py
```

The default input is `data/processed/fiqa_fast/`; override it with
`--input-dir PATH` if you used another preprocessing output. It writes
split-specific Hugging Face Arrow datasets to
`data/tokenized/fiqa_fast_minilm_l6_v2/`, plus `manifest.json` and
`exclusions.jsonl`. Set `model.max_length` in `configs/base.yaml` or pass
`--max-length N`. The tokenizer truncates only the passage, never pads, and
logs/rejects a pair if its query plus pair special tokens cannot fit. Dynamic
padding belongs in the later training data collator.

Labels are stored as nullable `float32`: train labels remain binary floats,
and unjudged evaluation candidates remain null. Each record carries its
query/document IDs, stable row-based pair ID, split, source row index,
judgment/provenance fields, and aligned `input_ids`, `attention_mask`, and
`token_type_ids`. The manifest records model revision, tokenizer options,
seed, package versions, split truncation counts/fractions, and output file
hashes. This step does not instantiate a training loop or change the TSVs.

The selected checkpoint is a reranker fine-tuned on MS MARCO Passage Ranking
and its model card declares Apache-2.0. FiQA is a different benchmark, but no
exact query/document overlap audit has been run between the two training
corpora; record this limitation when interpreting baseline results.

## Step 3: baseline and trainer scripts (not run automatically)

`configs/step3.yaml` contains the pinned model and FiQA qrels revisions,
tokenized data location, batch sizes, optimizer/scheduler settings, stopping
rules, device preference, and output paths. It prefers CUDA when PyTorch can
access an NVIDIA GPU, and explicitly reports CPU fallback if CUDA is
unavailable. The project pins `torch==2.9.1+cu126` in `pyproject.toml` and
`uv.lock`. Run the baseline first:

```powershell
uv run python scripts/score_baseline.py
```

It scores every validation and test candidate using the local pretrained model
and writes `validation_scores.tsv`, `test_scores.tsv`, `metrics.json`, and
`resolved_config.yaml` under `reports/baseline/`. Metrics are query-averaged
nDCG@10 and MRR@10 against the pinned original `BeIR/fiqa-qrels` snapshots.
Candidate ranking includes unjudged documents; their score is retained, while
their unknown judgment is never written as a fabricated label. As in ordinary
qrels-based retrieval metrics, candidates without a qrel contribute no gain;
the ideal DCG includes all positive qrels, even positives absent from the
retrieval candidate list.

The trainer is defined but is not launched by this project setup. When you
choose to run it:

```powershell
uv run python scripts/train_reranker.py
```

Each invocation writes to a new subdirectory under `outputs/step3/`; an
existing nonempty run directory is never reused. By default, the directory is
named with the local start time and a random animal, such as
`20261018_110322_otter`. Set a readable name explicitly with
`--training-run-name`:

```powershell
uv run python scripts/train_reranker.py --training-run-name fiqa-batch32-v1
```

The untouched-model validation baseline runs before training by default. Skip
that extra scoring pass with `--skip-baseline-validation`:

```powershell
uv run python scripts/train_reranker.py --skip-baseline-validation
```

Before updating weights, the trainer scores the untouched pretrained model on
the validation candidates and saves its scores and metrics under
`outputs/step3/pretraining_baseline/`. This uses the same candidates and qrels
as epoch validation, so the initial and fine-tuned validation metrics can be
compared directly; it does not score the test split.

It uses `BCEWithLogitsLoss` on float32 labels and excludes null/unjudged labels.
The collator dynamically pads batches (the current YAML config uses batch size
32). AdamW uses LR 2e-5 and weight decay 0.01, with bias and LayerNorm
parameters excluded from decay; the linear schedule warms up for 10%, gradients
are clipped at 1.0, and the current config allows at most 2 epochs. It evaluates validation nDCG@10 every epoch, saves
epoch checkpoints, selects/restores the best validation checkpoint, and stops
after two non-improving validation epochs. Test scoring happens once after
selection. Outputs include epoch scores, per-epoch checkpoints, best model,
history, final test scores/metrics, run metadata, and the resolved YAML config
under each run's directory in `outputs/step3/`. Mixed precision is enabled on
CUDA by default and can be disabled in the config.

The isolated overfit diagnostic is separate from regular training:

```powershell
uv run python scripts/train_reranker.py --overfit-one-batch
```

It performs the configured number of updates on one labeled batch, writes a
loss trace, and reloads/checks the original pretrained weights afterward. It
does not create or modify the training checkpoints. The Step 3 scripts and
tests have not been executed as part of authoring these instructions.

## Easy-only training experiment

The separate easy sampler preserves official train positives, samples four
uniform random corpus negatives per positive, and excludes every known positive
for that query across supplied official qrels. Corpus IDs are sorted once, and
each query has a SHA256-derived RNG seed. It samples all negatives for a query
without replacement, avoiding repeated query/document pairs even for queries
with multiple positives. It fails if the unique eligible pool cannot satisfy
the configured ratio. Random sampling does not guarantee a different topic or
irrelevance; unjudged negatives retain their false-negative-risk provenance.

Create the new dataset (the original preprocessing scripts are unchanged):

```bash
uv run python scripts/prepare_fiqa_data_easy.py
```

Settings are in `configs/data_easy.yaml`. Outputs are saved separately under
`data/processed/fiqa_easy/`: `train.tsv`, `validation.tsv`, `test.tsv`,
`exclusions.tsv`, and `manifest.json`. Named stages and progress bars show
loading/validation, sampling, export, and hashing; the final message lists full
output paths and prints a train preview. A nonempty output directory is refused;
use `--output-dir` to choose another one.

The existing `data/processed/fiqa_fast/manifest.json`, `validation.tsv`, and
`test.tsv` are required. Source revisions and checksums are verified, and the
evaluation TSVs are copied byte for byte, keeping candidate lists, null labels,
order, and original retrieval metadata. Evaluation is never sampled or balanced.
With the current retained train qrels, expect 14,131 positives + 56,524 easy
negatives = 70,655 train rows; validation and test remain 50,000 and 64,754 rows.

Tokenize separately with the same local model and length 256:

```bash
uv run python scripts/tokenize_fiqa_pairs.py --input-dir data/processed/fiqa_easy --output-dir data/tokenized/fiqa_easy_minilm_l6_v2 --max-length 256
```

When ready, train using the separate config and a fresh run name:

```bash
uv run python scripts/train_reranker.py --config configs/step3_easy.yaml --training-run-name easy-batch32-v1
```

This config keeps the model and optimizer settings from `configs/step3.yaml`,
changes the data locations, and writes runs under `outputs/step3_easy/`. The
initial validation baseline is still enabled. Select improvements on validation;
do not use the final test metric to choose training settings. This experiment
changes both negative source and duplicate-pair weighting, so any change cannot
be attributed to BM25 removal alone.

After tokenization, audit the new artifacts with:

```bash
uv run python scripts/audit_fiqa_data.py --config configs/step3_easy.yaml --output-dir reports/data_quality/easy
```

## Pretrained test evaluation

To evaluate only the untouched pretrained model on the easy experiment's test
candidates, without training or scoring validation:

```bash
uv run python scripts/score_baseline.py --config configs/step3_easy.yaml --split test
```

This loads the configured local pretrained snapshot and scores every tokenized
test candidate against the original pinned test qrels. It creates a timestamped
directory under `reports/baseline_easy/` containing `test_scores.tsv`,
`metrics.json`, and `resolved_config.yaml`. The final console lines print the
absolute output paths. Use `--output-dir PATH` to choose a new output directory;
an existing nonempty directory is rejected for these explicit scoring runs.
Compare these test metrics with the fine-tuned test metrics, rather than with
the validation baseline. Keep subsequent tuning decisions based on validation.

## Data quality audit

Run the reproducible audit across the processed and tokenized train, validation,
and test splits:

```powershell
uv run python scripts/audit_fiqa_data.py
```

It joins each saved `row_index`, query ID, and document ID to the processed TSV;
re-tokenizes the source query/passage with the pinned local tokenizer and saved
settings; checks labels against pinned official qrels; and reports passage
truncation by split, label, and negative provenance. It writes issue tables and
seeded review samples under `reports/data_quality/`. The negative review file
keeps random/easy, BM25/hard, fallback, and unjudged evaluation candidates
distinct. Unjudged candidates remain unknown and require human judgment. The
truncation sample export includes both the full passage and the decoded passage
tokens retained by the model.

The audit script has been run against the local data; the training scripts and
test suite have not been run as part of this audit.
