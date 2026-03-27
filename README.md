# G20 Movie Recommendation Project
This repo contains an end-to-end recommender-system experiment on MovieLens `1M` and `10M`.

It loads/unzips the MovieLens archives, standardizes them, builds corrected time-aware per-user train/validation/test splits (`80/10/10`), and evaluates models with an event-based top-N ranking protocol (including full-history negative exclusion).

Models trained and compared include: simple rating/bias baselines, item-item KNN, an ALS residual latent-factor model with separately estimated bias terms, and an optional NeuMF extension on `1M` only. The trained ALS artifacts are also used by a small CLI to generate recommendations for both existing-user and cold-start scenarios.

## Project Overview

The pipeline is organized as follows:

1. `scripts/prepare_data.py`: preprocess MovieLens, create time-aware splits, and build the ranking-evaluation candidate logic.
2. `scripts/baselines.py`: rating + ranking baselines (global mean / user/item means / bias terms and simple ranking fallbacks).
3. `scripts/collab_filtering.py`: collaborative filtering models (KNN and ALS variants).
4. `scripts/neumf.py`: NeuMF training/evaluation on `1M` only.
5. `scripts/ranking_eval.py`: event-based ranking metrics (Precision@K/Recall@K/NDCG@K) with negative exclusion.
6. `scripts/cli_recommend.py`: load trained ALS artifacts and print top-K recommendations for an existing or cold-start user.

## Setup

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Expected local archives in the repo root:

- `ml-1m.zip`
- `ml-10m.zip`

Download links:

- `1m`: https://files.grouplens.org/datasets/movielens/ml-1m.zip
- `10m`: https://files.grouplens.org/datasets/movielens/ml-10m.zip

## Fastest Way To Reproduce

- `make quick`: run the full pipeline on `MovieLens 1M`
- `make all`: run the full project pipeline on `MovieLens 1M` and `MovieLens 10M`
- `make cf-10m-resume`: resume the long `10m` collaborative-filtering run if it gets interrupted

Commands:

```bash
make quick
make all
make cf-10m-resume
```

## What The Make Targets Do

- `make prep`: regenerate processed `1m` and `10m` data
- `make baselines`: rerun baseline models on both datasets
- `make cf`: rerun KNN and ALS on both datasets
- `make neumf`: rerun NeuMF on `1m`
- `make quick`: `1m` prep + baselines + CF + NeuMF
- `make all`: `1m` and `10m` prep + baselines + CF, plus `1m` NeuMF
- `make test`: run unit tests

## Method Summary

The current pipeline uses:

- per-user time-ordered `80/10/10` train/validation/test splits
- full-history negative exclusion
- event-based sampled ranking evaluation
- final baseline, KNN, and ALS test metrics trained on `train + val`
- a default cap of `20,000` held-out events per dataset for ranking evaluation
- NeuMF training positives defined as ratings `>= 4.0`

This cap is a scalability tradeoff: ranking metrics are sampled estimates, not exhaustive full-dataset scores.

## Current Results Snapshot

- KNN is the strongest ranking model on both `1m` and `10m`
- the regularized `user_movie_bias` baseline is the best RMSE model on both `1m` and `10m`
- ALS remains the latent-factor model used for exportable CLI artifacts, not the headline ranking winner

## Outputs

Important outputs:

- processed data: `data/processed/<dataset>/`
- baseline summaries: `data/results/baselines/<dataset>/results.json`
- KNN/ALS summaries: `data/results/collab_filtering/<dataset>/results.json`
- NeuMF summary: `data/results/neumf/1m/results.json`
- ALS CLI artifacts: `data/models/als/<dataset>/als_residual/`

You can ignore `progress.json` files unless you are resuming a long run.

## CLI
Use one of these two ways to get recommendations.

### Option 1: Interactive Shell (easiest)
Start the guided shell:

```bash
python3 scripts/recommend_shell.py
```

Then answer prompts for:
- dataset (`1m` or `10m`)
- mode (`existing user` or `cold-start`)
- input (`userId` for existing users, or `movieId:rating,...` for cold-start)
- `top-k` and output format (`text` or `json`)

This is the recommended path for normal use because it avoids long commands and shows readable output by default.

### Option 2: Command Mode (scriptable)
Use direct commands when you want automation or exact repeatability.

Existing user:

```bash
python3 scripts/cli_recommend.py --dataset 10m --user 1 --top-k 10
```

Cold-start user:

```bash
python3 scripts/cli_recommend.py --dataset 10m --ratings "1:5,260:3.5,1193:4" --top-k 10
```

Common flags:
- `--dataset {1m|10m}`: choose dataset (matching processed data and trained artifacts must exist).
- `--user` / `--user-id`: existing MovieLens raw `userId` (mapped internally via `data/processed/<dataset>/user_map.csv`).
- `--ratings` / `--new-user-ratings`: cold-start input as raw `movieId:rating` pairs, e.g. `"1:5,260:3.5,1193:4"`.
  - `1:5` means movie `1` rated `5.0`.
  - Ratings should be on the MovieLens 0.5-5 scale.
- `--ratings-file <path>`: same as `--ratings`, but read from a file.
- `--top-k`: number of recommendations to print (default `10`).
- `--format {text|json}`: output format (default `text`).

Behavior notes:
- Existing-user mode excludes already-rated movies by default.
- Cold-start mode uses the provided ratings to infer preferences before ranking candidates.

## Tests

```bash
make test
```
