# Research Steps Log

This file is the report-style source of truth for things implemented in the project.

## Canonical Methodology

The current project uses one fixed evaluation policy:

- per-user time-aware `80/10/10` split
- oldest ratings go to train, next block to validation, newest block to test
- ranking candidates exclude the user's full known history
- ranking metrics are event-based, sampled, and averaged across held-out events
- final reported baseline, KNN, and ALS test metrics are produced after training on `train + val`
- ALS hyperparameter tuning uses validation metrics only; test metrics are computed only in the final stage
- NeuMF remains `1m` only and uses the same sampled event-based ranking protocol
- NeuMF training treats ratings `>= 4.0` as positive interactions for the neural ranking objective

This project does not use leave-last-two-out, user-averaged ranking metrics, or a configurable negative-scope policy.

## Current Result Framing

The current rerun should be described with the following conclusions:

- KNN is the strongest top-N ranking model on both `1m` and `10m`
- the regularized `user_movie_bias` baseline is the best RMSE model on both `1m` and `10m`
- ALS residual factorization is retained as the latent-factor comparison model and as the CLI artifact source, not as the assumed headline winner
- NeuMF is a `1m` neural comparison only; it is not yet part of the `10m` benchmark table

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
- split validation checks

### `make baselines`

- `scripts/baselines.py` for `1m`
- `scripts/baselines.py` for `10m`

Models:

- rating: `global_mean`, `user_mean`, `item_mean`, `user_movie_bias`
- ranking: `popularity`, `user_movie_bias_ranking`, `metadata_global`, `genre_tag_profile`

Final-fit policy:

- baselines are trained on `train + val` before final test evaluation

### `make cf`

- `scripts/collab_filtering.py` for `1m`
- `scripts/collab_filtering.py` for `10m`

Models:

- item-item KNN
- ALS residual model with separately estimated bias terms

Notes:

- ranking evaluation is event-based and sampled
- default ranking cap is `20,000` held-out events per dataset
- `10m` uses safer Spark defaults and finer ALS block settings
- ALS tuning uses validation metrics only
- final KNN and ALS models are trained on `train + val` for the saved test results and CLI artifacts
- residual ALS was chosen because Spark MLlib gives a stable explicit-feedback ALS solver directly; jointly biased ALS would require a custom training path and is future work rather than the current implementation

### `make neumf`

- `scripts/neumf.py` for `1m`

Notes:

- NeuMF is currently `1m` only
- validation and test ranking use the same event-based candidate logic
- training positives are limited to ratings `>= 4.0` so the neural objective better matches a "recommend likely-liked movies" interpretation

### `make test`

- unit tests under `tests/`

## CLI Notes

The CLI still uses ALS artifacts for the main recommendation flow.

Why ALS is used for the CLI instead of KNN:

- KNN is the stronger ranking model in the saved evaluation results
- ALS is still used for the CLI because its exported latent factors and bias terms make interactive scoring much simpler
- existing-user recommendations can be scored with one user-factor vector against all item factors
- cold-start recommendations are easier with ALS because a new user vector can be inferred directly from a few provided ratings
- a KNN-backed CLI would be possible, but it would require more stateful user-history logic and a less clean cold-start path

Cold-start behavior now has two paths:

- if enough mapped ratings are provided, infer a new ALS user vector
- if ratings are below `--cold-start-min-ratings`, use a metadata fallback based on genres, popularity, and available tags

Both paths exclude the provided rated movies from recommendations by default.

## Important Output Files

- processed metadata: `data/processed/<dataset>/meta.json`
- baseline summaries: `data/results/baselines/<dataset>/results.json`
- collaborative-filtering summaries: `data/results/collab_filtering/<dataset>/results.json`
- NeuMF summary: `data/results/neumf/1m/results.json`
- CLI ALS artifacts: `data/models/als/<dataset>/als_residual/`

## Status Note

Use `make quick` for a fast `1m` rerun, then `make all` for the full corrected project rerun.

## Future Work

- scale NeuMF to a sampled `10m` subset if more compute time is available
- implement a jointly biased ALS formulation or a ranking-optimized factor model such as BPR or implicit ALS if deeper model work is needed

## Proposal Revision Notes

These notes describe the main differences between the original proposal and the delivered project, along with why those changes were made.

- Split policy changed from leave-last-two-out wording to a per-user time-aware `80/10/10` split.
  Reason: the delivered project evaluates both rating prediction and top-N ranking, and a per-user `80/10/10` split gives substantially larger validation and test blocks than leave-last-two-out. That produces more stable RMSE estimates, more stable sampled ranking estimates, and more useful validation data for hyperparameter tuning. Leave-last-two-out is better if the only goal is a strict next-item setup, but the `80/10/10` split is a better fit for this project’s combined rating-and-ranking evaluation.

- Ranking evaluation changed from generic user-averaged wording to sampled event-based evaluation.
  Reason: the delivered project uses one held-out interaction plus sampled negatives because that is much more tractable at MovieLens scale and aligns cleanly with the candidate-generation pipeline used for all models. A full user-level relevant-set evaluation would require a separate relevance definition, much heavier scoring, and a different metric pipeline. For this project, sampled event-based evaluation is the more practical and internally consistent choice, even though it is a narrower definition of ranking quality than exhaustive user-level evaluation.

- ALS description changed from “ALS with biases” to ALS on residual ratings with separately estimated bias terms.
  Reason: Spark MLlib gives a stable explicit-feedback ALS solver, but not a turnkey jointly biased ALS implementation with the same ease of training and export. Estimating biases separately and fitting ALS on residuals preserves most of the intended behavior while keeping the system simpler, easier to debug, and easier to package for the CLI. A jointly trained biased factor model could be better in theory, but the residual formulation was the better engineering choice for the delivered project.

- The abstract and model framing were updated so ALS is no longer assumed to be the top-ranking winner.
  Reason: the rerun results show KNN is the strongest ranking model, so the delivered project should be described as a comparison whose winner is determined by the shared evaluation protocol rather than by the original modeling expectation.

- NeuMF scope changed from “scale up to 10M if feasible” language to explicit `1m` implementation with `10m` scale-up framed as future work.
  Reason: the neural model is the most expensive part of the project and was always secondary to the main baseline/KNN/ALS comparison. Running NeuMF on `1m` was enough to validate the neural extension, compare it against the other models, and keep the project manageable. Scaling NeuMF further would add cost and complexity without improving the core reproducibility of the main pipeline, so treating larger-scale NeuMF as future work was the better scope decision.

- Ranking metric definitions were rewritten for the one-positive-per-event candidate setup.
  Reason: this was not a methodological improvement so much as a clarity correction. The original proposal used generic recommender-metric definitions, but the delivered project uses event-level hit-style Precision@K and Recall@K under sampled candidate groups. Rewriting the definitions was necessary so the project description accurately matches what the code and saved results actually mean.
