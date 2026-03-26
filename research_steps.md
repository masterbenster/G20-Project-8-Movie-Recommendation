# Research Steps Log

This file is appended as we complete each step of the research plan.

## Step 1: Data Pipeline and Splits

### What we implemented

We implemented a reproducible Step 1 pipeline to:

1. Load MovieLens ratings from the provided zip files.
2. Clean and deduplicate ratings by keeping only the **latest** `(userId, movieId)` rating per user/movie pair.
3. Build a **time-aware per-user split**:
   - last interaction for each user -> `test`
   - second-last interaction for each user -> `val`
   - all earlier interactions -> `train`
   - users with fewer than 3 interactions are removed (so val/test both exist)
4. Build contiguous ID mappings for models:
   - `user_index` for `userId`
   - `movie_index` for `movieId`
5. Generate **negative samples** for ranking evaluation:
   - for each positive `(user_index, pos_movie_index)` in `val` and `test`, sample `99` movies the user **never interacted with in the training split only**.

### Code added
- `scripts/prepare_data.py`

### Commands run

```bash
python3 scripts/prepare_data.py --dataset 1m --num-negatives 99 --seed 42
python3 scripts/prepare_data.py --dataset 10m --num-negatives 99 --seed 42
```

### Outputs written

All outputs are placed under `data/processed/<dataset>/`.

For `MovieLens 1M` (`data/processed/1m/`):

- `ratings_clean.csv.gz`
- `train.csv.gz`, `val.csv.gz`, `test.csv.gz`
- `val_negs.csv.gz`, `test_negs.csv.gz`
- `user_map.csv`, `movie_map.csv`
- `meta.json`

For `MovieLens 10M` (`data/processed/10m/`):

- `ratings_clean.csv.gz`
- `train.csv.gz`, `val.csv.gz`, `test.csv.gz`
- `val_negs.csv.gz`, `test_negs.csv.gz`
- `user_map.csv`, `movie_map.csv`
- `meta.json`

### Dataset summary (from `meta.json`)

MovieLens 1M:

- `num_users`: 6040
- `num_movies`: 3706
- `num_ratings_clean`: 1000209
- `num_train`: 988129
- `num_val`: 6040
- `num_test`: 6040
- `num_val_negs_rows`: 597960
- `num_test_negs_rows`: 597960

MovieLens 10M:

- `num_users`: 69878
- `num_movies`: 10677
- `num_ratings_clean`: 10000054
- `num_train`: 9860298
- `num_val`: 69878
- `num_test`: 69878
- `num_val_negs_rows`: 6917922
- `num_test_negs_rows`: 6917922

### How to run again (if needed)

```bash
python3 scripts/prepare_data.py --dataset both --num-negatives 99 --seed 42
```

## Step 2: Baseline Models

### What we implemented
Step 2 implements the baseline models described in the project proposal:

1. Rating prediction (RMSE/MAE)
   - Global mean predictor.
   - User/movie bias model trained on `train.csv.gz` with regularization:
     `rating ~= mu + b_u[user] + b_i[item]`.

2. Top-N ranking (Precision@K, Recall@K, NDCG@K)
   - Popularity baseline: scores movies by `log1p(#ratings in train)`.
   - Genre/tag fallback: user-independent composite score using
     training popularity + genre information from `movies.dat` + tag frequency from `tags.dat` (when available).

All metrics are computed on the Step-1 `test` split using `test_negs.csv.gz`
(one positive item per user + sampled negatives).

### Code added
- `scripts/baselines.py`

### Commands run
```bash
python3 scripts/baselines.py --dataset 1m --ks 5,10,20 --lambda-reg 25 --num-iters 10
python3 scripts/baselines.py --dataset 10m --ks 5,10,20 --lambda-reg 25 --num-iters 10
```

### Outputs written
For each dataset, outputs were written to:
`data/results/baselines/<dataset>/`

Key files:
- `results.json` (all metrics)
- `bias_bu.npy`, `bias_bi.npy` (bias parameters)
- `popularity_scores.npy` (popularity baseline scores per `movie_index`)
- `genre_tag_scores.npy` (genre/tag fallback scores per `movie_index`)

You can inspect exact metric values in:
- `data/results/baselines/1m/results.json`
- `data/results/baselines/10m/results.json`

### Quick metric snapshot (from `results.json`)
MovieLens 1M:
- Rating prediction: `global_mean` RMSE=1.168, MAE=0.982; `user_movie_bias` RMSE=0.966, MAE=0.771
- Top-N (popularity) NDCG@10=0.240

MovieLens 10M:
- Rating prediction: `global_mean` RMSE=1.092, MAE=0.900; `user_movie_bias` RMSE=0.937, MAE=0.731
- Top-N (popularity) NDCG@10=0.396

## Step 3: Collaborative Filtering Models

### What we implemented
Step 3 implements the collaborative filtering models from the project proposal:

1. Item-item KNN collaborative filtering (cosine similarity)
   - Computes item-item cosine similarity from user ratings in training.
   - Predicts scores for a user/item by aggregating similarities to items the user rated.

2. ALS matrix factorization (Spark MLlib) with user/item bias terms
   - Approximates bias terms by training ALS on residuals:
     `rating_resid = rating - mu - b_u - b_i`.
   - Final prediction:
     `rating_pred = mu + b_u + b_i + ALS(u,i)`.

### Code added
- `scripts/collab_filtering.py`

### Command run (MovieLens 1M)
```bash
python3 scripts/collab_filtering.py \
  --dataset 1m --ks 5,10,20 --knn-neighbors 20 --als-ranks 20 --als-regs 0.1 --als-max-iter 8 --als-bias-lambda 25 --seed 42
```

### Outputs written
- `data/results/collab_filtering/1m/results.json`

### Quick metric snapshot (from `data/results/collab_filtering/1m/results.json`)
MovieLens 1M:
- Item-item KNN: test rating RMSE=1.079, MAE=0.813; test NDCG@10=0.386
- ALS (residual + biases): test rating RMSE=0.921, MAE=0.727; test NDCG@10=0.113

### Command run (MovieLens 10M)
```bash
python3 scripts/collab_filtering.py \
  --dataset 10m --resume --ks 5,10,20 --knn-neighbors 50 --als-ranks 20 --als-regs 0.1 --als-max-iter 8 --als-bias-lambda 25 --seed 42
```

### Outputs written
- `data/results/collab_filtering/10m/results.json`

### Quick metric snapshot (from `data/results/collab_filtering/10m/results.json`)
MovieLens 10M:
- Item-item KNN (neighbor_k=50): test rating RMSE=0.990, MAE=0.741; test NDCG@10=0.572
- ALS (rank=20, reg=0.1): test rating RMSE=0.891, MAE=0.689; test NDCG@10=0.161

## Step 4: Neural Extension (NeuMF)

### What we implemented
Step 4 implements a NeuMF (Neural Collaborative Filtering) model in PyTorch:
- User/item embeddings (GMF component + MLP component)
- Trained using explicit positive interactions from `train.csv.gz` plus sampled negatives
- Evaluated using the Step-1 ranking candidate sets in `val_negs.csv.gz` and `test_negs.csv.gz`

### Code added
- `scripts/neumf.py`

### Command run (MovieLens 1M)
```bash
python3 scripts/neumf.py \
  --dataset 1m --epochs 6 --batch-size 2048 --lr 0.001 --weight-decay 1e-6 \
  --neg-ratio 1 --embed-dim 32 --mlp 64,32,16 --dropout 0.2 --seed 42 \
  --ks 5,10,20
```

### Outputs written
- `data/results/neumf/1m/results.json`
- `data/results/neumf/1m/model.pt`

Note: test evaluation reloads the best validation checkpoint weights before computing `test_metrics`.

### Quick metric snapshot (from `results.json`)
MovieLens 1M NeuMF:
- Validation NDCG@10=0.359
- Test NDCG@10=0.339
- Test top-N:
  - Precision@10=0.0602
  - Recall@10=0.6023

## Step 5: Packaging, Demo, and Report

### What we implemented
1. Exported ALS artifacts (latent factors + biases) after Step 3 so we can do fast top-N inference without re-training.
2. Implemented a command line demo that loads the exported ALS artifacts and prints top-N recommended movies for a user.

### Code added
- `scripts/cli_recommend.py`

### ALS artifacts used by the CLI
For MovieLens 10M (exported from Step 3):
- `data/models/als/10m/als_residual/mu.npy`
- `data/models/als/10m/als_residual/bias_bu.npy`
- `data/models/als/10m/als_residual/bias_bi.npy`
- `data/models/als/10m/als_residual/user_factors.npy`
- `data/models/als/10m/als_residual/item_factors.npy`
- `data/models/als/10m/als_residual/meta.json`

### Command run (demo)
```bash
python3 scripts/cli_recommend.py --dataset 10m --user-id 1 --top-k 10
```

### Outputs written / displayed
The CLI prints ranked recommendations (movie title + genres) to stdout for the chosen user.

### Cold-start CLI support (new-user ratings)

We extended `scripts/cli_recommend.py` to support a basic cold-start demo: a brand-new user provides a few MovieLens ratings, and the CLI infers a compatible “user vector” from the exported ALS artifacts (no re-training).

#### What changed
- Added `--new-user-ratings` mode for new users.
- Kept the existing `--user-id` / `--user-index` mode for existing MovieLens users.
- Added optional cold-start controls:
  - `--cold-start-reg` (ridge regularization strength; default uses `reg_param` from `meta.json` if present)
  - `--cold-start-min-ratings` (warning if fewer than this many mapped ratings are provided)

#### CLI usage
- Existing user (unchanged):
```bash
python3 scripts/cli_recommend.py --dataset 10m --user-id 1 --top-k 10
```
- New user cold-start:
```bash
python3 scripts/cli_recommend.py --dataset 10m \
  --new-user-ratings "1:5,260:3.5,1193:4" \
  --top-k 10
```

#### Input format
- `--new-user-ratings` expects comma-separated `movieId:rating` pairs.
- Movie ids are MovieLens raw `movieId` values (as in the original MovieLens ratings files).

#### How cold-start works (high level)
- The CLI maps each provided MovieLens `movieId` to the internal contiguous `movie_index` using `data/processed/<dataset>/movie_map.csv`.
- Using the exported ALS parameters, it estimates new user parameters from the few `(movie_index, rating)` points.
- It scores all movies for the inferred user and prints the top-N recommendations.
- By default, it excludes the movies included in the cold-start input (`--exclude-rated` is on by default).

#### Artifacts required
- ALS residual artifacts under `data/models/als/<dataset>/als_residual/`:
  - `mu.npy`, `bias_bi.npy`, `item_factors.npy`
  - (optionally `meta.json` for default `reg_param`)
- Mapping file under `data/processed/<dataset>/movie_map.csv` for raw `movieId` -> `movie_index`.
