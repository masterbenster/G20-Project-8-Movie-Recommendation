# G20 Movie Recommendation Project
This repo contains an end-to-end recommender-system experiment on MovieLens `1M` and `10M`.

It loads/unzips the MovieLens archives, standardizes them, builds corrected time-aware per-user train/validation/test splits (`80/10/10`), and evaluates models with an event-based top-N ranking protocol (including full-history negative exclusion).

Models trained and compared include: simple rating/bias baselines, item-item KNN, ALS matrix factorization (main collaborative-filtering model), and an optional NeuMF extension on `1M` only. The trained ALS artifacts are also used by a small CLI to generate recommendations for both existing-user and cold-start scenarios.

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
- full-history negative exclusion by default (`negative_scope=all`)
- on-the-fly ranking candidate generation instead of giant negative CSVs
- a default cap of `20,000` held-out events per dataset for ranking evaluation

This cap is a scalability tradeoff: ranking metrics are sampled estimates, not exhaustive full-dataset scores.

## Outputs

Important outputs:

- processed data: `data/processed/<dataset>/`
- baseline summaries: `data/results/baselines/<dataset>/results.json`
- KNN/ALS summaries: `data/results/collab_filtering/<dataset>/results.json`
- NeuMF summary: `data/results/neumf/1m/results.json`
- ALS CLI artifacts: `data/models/als/<dataset>/als_residual/`

You can ignore `progress.json` files unless you are resuming a long run.

## CLI
The CLI prints top-`K` recommended movies using the trained ALS residual factors (from `data/models/als/<dataset>/als_residual/`).

### Interactive Shell (Recommended)
If you want a short, repeated workflow (no long flags), start:

```bash
python3 scripts/recommend_shell.py
```

You will be prompted for:
- dataset (`1m` or `10m`)
- mode: existing user vs cold-start
- input (a raw `userId` for existing users, or `movieId:rating,...` for cold-start)
- `top-k` and output format (`text` or `json`)

The shell formats recommendations readably (title + score + genres) so you can skim results quickly.

Flags used in the examples:
- `--dataset {1m|10m}`: select which MovieLens variant to load (the corresponding ALS artifacts must exist).
- `--user-id` (alias: `--user`): an existing user's *raw* MovieLens `userId` (not an internal contiguous index). The script maps it via `data/processed/<dataset>/user_map.csv`.
- `--ratings` (alias: `--new-user-ratings`): cold-start mode. A quoted, comma-separated list of `movieId:rating` pairs using *raw* MovieLens `movieId`s (example format: `"1:5,260:3.5,1193:4"`).
  - Each `movieId:rating` pair means: user rated the MovieLens movie with id `movieId` as `rating`.
  - In the example: `1:5` means `(movieId=1, rating=5.0)`, `260:3.5` means `(movieId=260, rating=3.5)`, and `1193:4` means `(movieId=1193, rating=4.0)`.
  - Ratings should be on the MovieLens 0.5–5 scale.
- `--ratings-file <path>`: alternative to `--ratings`. A text file containing `movieId:rating` pairs (comma- or newline-separated).
- `--top-k`: how many movies to print (default `10`).
- `--format {text|json}`: output format (default `text`). Use `json` for machine-readable results.

Existing user:

```bash
python3 scripts/cli_recommend.py --dataset 10m --user 1 --top-k 10
```

Cold-start user:

```bash
python3 scripts/cli_recommend.py --dataset 10m --ratings "1:5,260:3.5,1193:4" --top-k 10
```

Notes:
- By default (`--format text`), the CLI prints a compact ranked list: `rank. title | score | genres`.
- Use `--format json` if you want structured output (including `movieId_raw` when available).
- For cold-start, the script infers a new user latent vector from the provided `movieId:rating` pairs (then excludes those provided movies from recommendations by default).
- For existing users, recommendations exclude movies the user has already rated (by default it excludes from the full processed history via `--exclude-seen-scope all`).

## Tests

```bash
make test
```

