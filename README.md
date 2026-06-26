# How Should We Explain Pairwise Reranking Decisions?

Explaining Pairwise Ranking Decisions in Neural Rerankers

This repository contains the code and final outputs for my MSc thesis. The project evaluates pointwise and pairwise feature attribution explanations for neural rerankers, comparing an independently scored cross-encoder with the natively pairwise DuoT5 model under faithfulness, stability, and token-agreement analyses.

The two rerankers used in the final experiments are:

- `cross-encoder/ms-marco-MiniLM-L-6-v2`
- `castorini/duot5-base-msmarco`

The main evaluation in the thesis is based on:

- faithfulness
- stability
- token agreement

## 1. Repository overview

Main folders:

- `scripts/`  
  Final scripts for reranking, attribution, evaluation, and plotting.
- `thesis_runs/`  
  Final outputs used in the thesis.
- `data/`  
  Local data directory expected by the scripts. This directory is not tracked in the GitHub repository. The final experiments use MS MARCO passage dev data, which is not included in this repository and should be downloaded separately from the official MS MARCO release (https://microsoft.github.io/msmarco/).
  The scripts expect these files:
  - `data/msmarco_passage_dev/raw/queries.dev.tsv`
  - `data/msmarco_passage_dev/raw/qrels.dev.tsv`
  - `data/msmarco_passage_dev/raw/top1000.dev`
- `notebooks/`  
  A few smaller exploratory notebooks used for EDA and initial experiments.


## 2. Reproducing the thesis results

The main thesis output summaries are already saved in `thesis_runs/`. To reproduce the full pipeline from scratch, run the scripts from the repository root in this order:

1. freeze query and explanation subsets (`scripts/freeze_query_scopes.py`, `scripts/freeze_explanation_subsets.py`, `scripts/freeze_stability_subset.py`)
2. run reranking (`scripts/run_cross_encoder_reranking_final.py`, `scripts/run_monot5_reranking_for_duot5_final.py`, `scripts/run_duot5_reranking_final.py`)
3. run attribution methods (`scripts/run_cross_encoder_ig_final.py`, `scripts/run_cross_encoder_loo_final.py`, `scripts/run_duot5_pairwise_ig_final.py`, `scripts/run_duot5_pointwise_ig_final.py`, `scripts/run_duot5_loo_final.py`)
4. run faithfulness, stability, and agreement evaluation (`scripts/run_faithfulness_final.py`, `scripts/run_sentence_retrieval_faithfulness.py`, `scripts/run_stability_final.py`, `scripts/run_explanation_agreement.py`, `scripts/run_faithfulness_significance_tests.py`)
5. generate tables and figures (`scripts/summarize_faithfulness_auc.py`, `scripts/plot_faithfulness_results.py`, `scripts/plot_stability_alternatives.py`, `scripts/plot_explanation_agreement_heatmaps.py`, `scripts/plot_qualitative_token_heatmaps.py`, `scripts/plot_sentence_retrieval_faithfulness.py`, `scripts/plot_retrieval_faithfulness_results.py`)


## 3. Environment setup

I ran the final scripts from the local virtual environment in `ig-thesis/`. A recent Python 3.9+ environment is recommended.

All commands below assume they are run from the repository root.

A simple setup would be:

```bash
python3 -m venv ig-thesis
source ig-thesis/bin/activate
pip install pandas numpy scipy matplotlib torch transformers sentence-transformers
```

Main packages used:

- `pandas`
- `numpy`
- `scipy`
- `matplotlib`
- `torch`
- `transformers`
- `sentence-transformers`

Some scripts will use Apple `mps` automatically if it is available.

To activate the environment later:

```bash
source ig-thesis/bin/activate
```


## 4. Model checkpoints used

These are the checkpoints used in the final pipeline:

- cross-encoder reranker:
  - `cross-encoder/ms-marco-MiniLM-L-6-v2`
- monoT5 first-stage reranker for DuoT5:
  - `castorini/monot5-base-msmarco`
- DuoT5 reranker:
  - `castorini/duot5-base-msmarco`

These names are also hard-coded in the final scripts.

## 5. Running reranking

### Cross-encoder

First freeze the query scope:

```bash
./ig-thesis/bin/python scripts/freeze_query_scopes.py
```

Then run the final cross-encoder reranking:

```bash
./ig-thesis/bin/python scripts/run_cross_encoder_reranking_final.py
```

Outputs go to:

- `thesis_runs/cross_encoder/reranking/`

These reranking outputs are generated locally and are not tracked in the GitHub repository because the full files are too large for regular git hosting.

### DuoT5

First run monoT5 on the frozen DuoT5 query scope:

```bash
./ig-thesis/bin/python scripts/run_monot5_reranking_for_duot5_final.py
```

Then run DuoT5 reranking on the monoT5 candidate pool:

```bash
./ig-thesis/bin/python scripts/run_duot5_reranking_final.py
```

Outputs go to:

- `thesis_runs/duot5/reranking/monot5_first_stage/`
- `thesis_runs/duot5/reranking/duot5_final/`

As with the cross-encoder reranking outputs, these files are kept locally and are not included in the GitHub repository.

## 6. Running attribution methods

Before running the explanation methods, freeze the explanation subset:

```bash
./ig-thesis/bin/python scripts/freeze_explanation_subsets.py
```

### Cross-encoder explanations

Integrated Gradients:

```bash
./ig-thesis/bin/python scripts/run_cross_encoder_ig_final.py
```

Leave-One-Out:

```bash
./ig-thesis/bin/python scripts/run_cross_encoder_loo_final.py
```

### DuoT5 explanations

Pairwise Integrated Gradients:

```bash
./ig-thesis/bin/python scripts/run_duot5_pairwise_ig_final.py
```

Pointwise proxy Integrated Gradients:

```bash
./ig-thesis/bin/python scripts/run_duot5_pointwise_ig_final.py
```

Leave-One-Out:

```bash
./ig-thesis/bin/python scripts/run_duot5_loo_final.py
```

Explanation outputs go to:

- `thesis_runs/cross_encoder/explanations/`
- `thesis_runs/duot5/explanations/`

Some attribution scripts, especially DuoT5 Integrated Gradients, can be slow and computationally expensive. The final outputs are included in `thesis_runs/` so the thesis results can be inspected without rerunning all attribution scripts.

## 7. Running faithfulness evaluation

Main perturbation-based faithfulness:

```bash
./ig-thesis/bin/python scripts/run_faithfulness_final.py --experiment all
```

Sentence-based retrieval faithfulness:

```bash
./ig-thesis/bin/python scripts/run_sentence_retrieval_faithfulness.py
```

I also kept a few older retrieval variants in the repo:

```bash
./ig-thesis/bin/python scripts/run_retrieval_faithfulness_final.py
./ig-thesis/bin/python scripts/run_window_retrieval_faithfulness.py
```

Faithfulness outputs go to:

- `thesis_runs/cross_encoder/faithfulness/`
- `thesis_runs/duot5/faithfulness/`

## 8. Running stability evaluation

Freeze the stability subset first:

```bash
./ig-thesis/bin/python scripts/freeze_stability_subset.py
```

Then run stability evaluation:

```bash
./ig-thesis/bin/python scripts/run_stability_final.py --experiment all
```

These runs can be relatively slow because they require recomputing explanations for perturbed inputs. The saved outputs in thesis_runs/ can be used directly if rerunning is not necessary.

Outputs go to:

- `thesis_runs/cross_encoder/stability/`
- `thesis_runs/duot5/stability/`

## 9. Generating tables and figures

Faithfulness plots:

```bash
./ig-thesis/bin/python scripts/plot_faithfulness_results.py
```

Stability plots:

```bash
./ig-thesis/bin/python scripts/plot_stability_alternatives.py
```

Agreement results and heatmaps:

```bash
./ig-thesis/bin/python scripts/run_explanation_agreement.py --experiment all
./ig-thesis/bin/python scripts/plot_explanation_agreement_heatmaps.py
```

Qualitative token heatmaps:

```bash
./ig-thesis/bin/python scripts/plot_qualitative_token_heatmaps.py
```

Sentence retrieval plots:

```bash
./ig-thesis/bin/python scripts/plot_sentence_retrieval_faithfulness.py
```

General retrieval plots:

```bash
./ig-thesis/bin/python scripts/plot_retrieval_faithfulness_results.py
```

Faithfulness AUC summary:

```bash
./ig-thesis/bin/python scripts/summarize_faithfulness_auc.py
```

Faithfulness significance tests:

```bash
./ig-thesis/bin/python scripts/run_faithfulness_significance_tests.py
```

## 10. Random seeds and frozen subsets

The final thesis results use fixed random seeds and frozen subsets so that the comparisons stay aligned across methods.

Important files:

- query scope:
  - `thesis_runs/shared/pair_definitions/duot5_queries_150_seed42_top100.csv`
  - `thesis_runs/shared/pair_definitions/cross_encoder_queries_full_eligible_top100.csv`
- explanation subset:
  - `thesis_runs/duot5/explanations/explanation_pairs_1_per_query_seed42.csv`
  - `thesis_runs/cross_encoder/explanations/explanation_pairs_1_per_query_on_duot5_queries_seed42.csv`
- stability subset:
  - `thesis_runs/cross_encoder/stability/stability_pairs_50_queries_seed42.csv`
  - `thesis_runs/duot5/stability/stability_pairs_50_queries_seed42.csv`

Metadata files:

- `thesis_runs/shared/pair_definitions/query_scope_info.json`
- `thesis_runs/shared/pair_definitions/explanation_subset_info.json`
- `thesis_runs/shared/pair_definitions/stability_subset_info.csv`

The main seed used in the final pipeline is `42`.

## 11. Saved results

The final thesis outputs are already saved in `thesis_runs/`.

Some useful files:

### Explanations

- `thesis_runs/cross_encoder/explanations/attributions_ig_summary.csv`
- `thesis_runs/cross_encoder/explanations/attributions_loo_summary.csv`
- `thesis_runs/duot5/explanations/attributions_pairwise_ig_summary.csv`
- `thesis_runs/duot5/explanations/attributions_pointwise_ig_summary.csv`
- `thesis_runs/duot5/explanations/attributions_loo_summary.csv`

### Faithfulness

- `thesis_runs/cross_encoder/faithfulness/faithfulness_results.csv`
- `thesis_runs/cross_encoder/faithfulness/faithfulness_summary.csv`
- `thesis_runs/cross_encoder/faithfulness/faithfulness_auc_summary.csv`
- `thesis_runs/duot5/faithfulness/faithfulness_results.csv`
- `thesis_runs/duot5/faithfulness/faithfulness_summary.csv`
- `thesis_runs/duot5/faithfulness/faithfulness_auc_summary.csv`

### Stability

- `thesis_runs/cross_encoder/stability/stability_summary.csv`
- `thesis_runs/duot5/stability/stability_summary.csv`

### Agreement

- `thesis_runs/cross_encoder/faithfulness/explanation_agreement_summary.csv`
- `thesis_runs/duot5/faithfulness/explanation_agreement_summary.csv`

### Other outputs

- `thesis_runs/shared/faithfulness_significance_summary.csv`
- `thesis_runs/shared/faithfulness_normality_checks.csv`
- `thesis_runs/shared/pair_definitions/`

The full reranking files and large attribution dumps are kept locally and can be regenerated with the scripts above if needed.

## Notes

This repository includes both final thesis outputs and exploratory files created during development. For the final results reported in the thesis, use `thesis_runs/` as the primary directory. The repository contains the code and outputs used to produce the final thesis results.
