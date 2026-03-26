# G20 Movie Recommendation Project

Movie recommendation system experiments on MovieLens 1M and 10M using:

- data preparation with time-aware per-user splits
- baseline mean/bias and metadata models
- item-item KNN collaborative filtering
- ALS matrix factorization in Spark
- NeuMF on MovieLens 1M
- a CLI for ALS recommendations and cold-start inference

## Setup

```bash
python3 -m pip install -r requirements.txt
```

Expected local archives in the repo root:

- `ml-1m.zip`
- `ml-10m.zip`

## Data Preparation

The corrected pipeline now uses:

- per-user time-ordered `80/10/10` train/validation/test splits
- full-history negative sampling by default (`negative_scope=all`)
- validation checks that reject seen-item negatives and split overlap
- on-the-fly ranking candidate generation instead of writing giant negative CSVs
- a default cap of `20,000` held-out events per dataset for ranking evaluation

Run:

```bash
python3 scripts/prepare_data.py --dataset 1m --num-negatives 99 --seed 42
python3 scripts/prepare_data.py --dataset 10m --num-negatives 99 --seed 42
```

Outputs are written under `data/processed/<dataset>/`.

## Baselines

Implemented rating baselines:

- `global_mean`
- `user_mean`
- `item_mean`
- `user_movie_bias`

Implemented ranking baselines:

- `popularity`
- `user_movie_bias_ranking`
- `metadata_global`
- `genre_tag_profile`

Run:

```bash
python3 scripts/baselines.py --dataset 1m --ks 5,10,20 --lambda-reg 25 --num-iters 10
python3 scripts/baselines.py --dataset 10m --ks 5,10,20 --lambda-reg 25 --num-iters 10
```

## Collaborative Filtering

Run KNN and ALS:

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

For 10M, rerun with the same flags and use `--resume` for long runs.

Notes:

- tuning-time validation metrics are stored separately from final test metrics
- final models no longer report leaked validation metrics from `train+val` retraining
- Spark defaults are tuned for this repo (`local[4]`, `4g` driver/executor memory, finer ALS blocks on `10m`)

## NeuMF

NeuMF currently targets MovieLens 1M.

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

## CLI

Existing user:

```bash
python3 scripts/cli_recommend.py --dataset 10m --user-id 1 --top-k 10
```

Cold-start user:

```bash
python3 scripts/cli_recommend.py \
  --dataset 10m \
  --new-user-ratings "1:5,260:3.5,1193:4" \
  --top-k 10
```

Seen-item filtering defaults to full history for existing users. Use:

```bash
python3 scripts/cli_recommend.py --dataset 10m --user-id 1 --exclude-seen-scope train
```

to emulate the old train-only behavior.

## Tests

```bash
python3 -m unittest discover -s tests -q
```

## Important Note

The methodology changed materially from the earlier repo state. Any ranking metrics or processed artifacts generated before these fixes should be treated as stale and regenerated before final reporting.
