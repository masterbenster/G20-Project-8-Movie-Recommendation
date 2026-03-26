# Research Steps Log

This file documents the simplified reproduction flow for the current corrected pipeline.

## Official Reproduction Commands

Quick reproducibility path:

```bash
make quick
```

Full rerun path:

```bash
make all
```

If the long `10m` collaborative-filtering run is interrupted:

```bash
make cf-10m-resume
```

## What Each Target Runs

### `make prep`

- `scripts/prepare_data.py` for `1m`
- `scripts/prepare_data.py` for `10m`

Method:

- deduplicate by latest `(user, movie)` rating
- time-aware per-user `80/10/10` split
- full-history negative exclusion
- on-the-fly ranking candidate generation
- split and candidate validation checks

### `make baselines`

- `scripts/baselines.py` for `1m`
- `scripts/baselines.py` for `10m`

Models:

- rating: `global_mean`, `user_mean`, `item_mean`, `user_movie_bias`
- ranking: `popularity`, `user_movie_bias_ranking`, `metadata_global`, `genre_tag_profile`

### `make cf`

- `scripts/collab_filtering.py` for `1m`
- `scripts/collab_filtering.py` for `10m`

Models:

- item-item KNN
- ALS residual model with bias terms

Notes:

- ranking evaluation is event-based and sampled
- default ranking cap is `20,000` held-out events per dataset
- `10m` uses safer Spark defaults and finer ALS block settings
- tuning validation metrics are kept separate from final test metrics

### `make neumf`

- `scripts/neumf.py` for `1m`

Notes:

- NeuMF is currently `1m` only
- validation and test ranking use the same event-based candidate logic

### `make test`

- unit tests under `tests/`

## Important Output Files

- processed metadata: `data/processed/<dataset>/meta.json`
- baseline summaries: `data/results/baselines/<dataset>/results.json`
- collaborative-filtering summaries: `data/results/collab_filtering/<dataset>/results.json`
- NeuMF summary: `data/results/neumf/1m/results.json`
- CLI ALS artifacts: `data/models/als/<dataset>/als_residual/`

## Status Note

The earlier repo state used a different split strategy and invalid ranking negatives. Anything generated before the corrected pipeline should be treated as stale.
