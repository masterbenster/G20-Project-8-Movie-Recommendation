# Research Steps Log

This file tracks the current methodology and rerun instructions for the project code in this repo.

## Current Methodology

### Step 1: Data Pipeline and Splits

Implemented in `scripts/prepare_data.py`:

1. Load MovieLens ratings from the local zip archives.
2. Deduplicate `(userId, movieId)` pairs by keeping the latest timestamped rating.
3. Build a time-aware per-user `80/10/10` split:
   - oldest 80% -> `train`
   - next 10% -> `val`
   - most recent 10% -> `test`
4. Build contiguous `user_index` and `movie_index` mappings.
5. Evaluate ranking on sampled held-out events with on-the-fly negatives from full user history.
6. Validate:
   - no train/val/test overlap
   - no seen-item negatives
   - consistent negatives-per-event for generated candidate groups

Rerun:

```bash
python3 scripts/prepare_data.py --dataset 1m --num-negatives 99 --seed 42
python3 scripts/prepare_data.py --dataset 10m --num-negatives 99 --seed 42
```

### Step 2: Baseline Models

Implemented in `scripts/baselines.py`:

- rating baselines:
  - `global_mean`
  - `user_mean`
  - `item_mean`
  - `user_movie_bias`
- ranking baselines:
  - `popularity`
  - `user_movie_bias_ranking`
  - `metadata_global`
  - `genre_tag_profile`

Rerun:

```bash
python3 scripts/baselines.py --dataset 1m --ks 5,10,20 --lambda-reg 25 --num-iters 10
python3 scripts/baselines.py --dataset 10m --ks 5,10,20 --lambda-reg 25 --num-iters 10
```

### Step 3: Collaborative Filtering Models

Implemented in `scripts/collab_filtering.py`:

- item-item KNN with cosine similarity
- ALS residual model with bias terms
- event-based ranking evaluation keyed by held-out positive event
- on-the-fly candidate generation capped at 20,000 held-out events by default
- clean separation between tuning validation metrics and final test metrics

Rerun:

```bash
python3 scripts/collab_filtering.py \
  --dataset 1m \
  --ks 5,10,20 \
  --knn-neighbors 20,50 \
  --als-ranks 10,20 \
  --als-regs 0.1,1.0 \
  --als-max-iter 8 \
  --als-bias-lambda 25 \
  --seed 42
```

Use the same command shape for `10m`, typically with `--resume`.

### Step 4: NeuMF

Implemented in `scripts/neumf.py`:

- NeuMF on MovieLens 1M
- train negative sampling from train history
- event-based ranking evaluation for validation and test

Rerun:

```bash
python3 scripts/neumf.py \
  --dataset 1m \
  --epochs 6 \
  --batch-size 2048 \
  --lr 0.001 \
  --weight-decay 1e-6 \
  --neg-ratio 1 \
  --embed-dim 32 \
  --mlp 64,32,16 \
  --dropout 0.2 \
  --seed 42 \
  --ks 5,10,20
```

### Step 5: Packaging and Demo

Implemented in `scripts/cli_recommend.py`:

- existing-user ALS recommendations
- cold-start ALS user-vector inference from seed ratings
- seen-item exclusion with `--exclude-seen-scope {train,all}`
- corrected title decoding for MovieLens metadata

Example commands:

```bash
python3 scripts/cli_recommend.py --dataset 10m --user-id 1 --top-k 10
python3 scripts/cli_recommend.py --dataset 10m --new-user-ratings "1:5,260:3.5,1193:4" --top-k 10
```

## Status Note

The earlier repo state used a different split strategy and stale ranking negatives. Any previously generated ranking metrics should be considered invalid for final reporting until the full experiment suite is rerun with the current pipeline.
