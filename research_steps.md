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
python3 /scripts/prepare_data.py --dataset 1m --num-negatives 99 --seed 42
python3 /scripts/prepare_data.py --dataset 10m --num-negatives 99 --seed 42
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

### How to rerun later (if needed)

```bash
python3 /scripts/prepare_data.py --dataset both --num-negatives 99 --seed 42
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
python3 /scripts/baselines.py --dataset 1m --ks 5,10,20 --lambda-reg 25 --num-iters 10
python3 /scripts/baselines.py --dataset 10m --ks 5,10,20 --lambda-reg 25 --num-iters 10
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
python3 /scripts/collab_filtering.py \
  --dataset 1m --ks 5,10,20 --knn-neighbors 20 --als-ranks 20 --als-regs 0.1 --als-max-iter 8 --als-bias-lambda 25 --seed 42
```

### Outputs written
- `data/results/collab_filtering/1m/results.json`

### Quick metric snapshot (from `data/results/collab_filtering/1m/results.json`)
MovieLens 1M:
- Item-item KNN: test rating RMSE=1.079, MAE=0.813; test NDCG@10=0.386
- ALS (residual + biases): test rating RMSE=0.921, MAE=0.727; test NDCG@10=0.113

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
python3 /scripts/neumf.py \
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
