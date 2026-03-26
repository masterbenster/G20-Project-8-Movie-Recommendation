# G20 Movie Recommendation Project

Movie recommendation experiments on MovieLens `1M` and `10M` with:

- corrected per-user time-aware `80/10/10` splits
- full-history negative exclusion for ranking evaluation
- baselines, KNN, ALS, and NeuMF (`1m` only)
- an ALS CLI for existing-user and cold-start recommendations

## Setup

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Expected local archives in the repo root:

- `ml-1m.zip`
- `ml-10m.zip`

## Fastest Way To Reproduce

For a quick end-to-end rerun on the smaller dataset:

```bash
make quick
```

For the full rerun used for the refreshed results:

```bash
make all
```

If the long `10m` collaborative-filtering run gets interrupted:

```bash
make cf-10m-resume
```

You can also use:

```bash
scripts/run_all.sh quick
scripts/run_all.sh all
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

Existing user:

```bash
python3 scripts/cli_recommend.py --dataset 10m --user-id 1 --top-k 10
```

Cold-start user:

```bash
python3 scripts/cli_recommend.py --dataset 10m --new-user-ratings "1:5,260:3.5,1193:4" --top-k 10
```

## Tests

```bash
make test
```

## Note

Any metrics or artifacts produced before the corrected split and ranking-evaluation changes should be treated as stale.
