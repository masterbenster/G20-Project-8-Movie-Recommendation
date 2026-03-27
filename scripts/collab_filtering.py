import argparse
import json
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

try:
    from scripts.ranking_eval import build_candidate_groups, default_max_ranking_events, read_csv_gz
except ModuleNotFoundError:
    from ranking_eval import build_candidate_groups, default_max_ranking_events, read_csv_gz


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = DATA_DIR / "results" / "collab_filtering"
MODELS_DIR = DATA_DIR / "models" / "als"


def _rmse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(np.square(err))))
    mae = float(np.mean(np.abs(err)))
    return rmse, mae


def _metrics_from_scores_onepos(
    event_ids: np.ndarray,
    pos_movie_indices: np.ndarray,
    movie_indices_for_scores: np.ndarray,
    scores_for_scores: np.ndarray,
    ks: list[int],
    *,
    tie_break: str = "movie_desc",
) -> dict[str, float]:
    """
    Compute Precision@K, Recall@K, NDCG@K when for each user:
    - there is exactly one positive movie id (pos_movie_indices)
    - scores_for_scores contains predictions for all movies in the candidate set
      excluding the positive OR including it depending on construction.

    We expect a specific construction:
    - For each held-out event, there are M candidate movies (pos + negatives) and we have:
      event_ids length = num_events * M
      movie_indices_for_scores includes the positive movie exactly once per user.
    - We assume score arrays correspond 1:1 with movie_indices_for_scores.
    """
    ks = sorted(ks)

    # Ensure stable grouping by event id.
    order_u = np.argsort(event_ids, kind="stable")
    event_ids = event_ids[order_u]
    pos_movie_indices = pos_movie_indices[order_u]
    movie_indices_for_scores = movie_indices_for_scores[order_u]
    scores_for_scores = scores_for_scores[order_u]

    # Group boundaries
    boundaries = np.flatnonzero(event_ids[1:] != event_ids[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(event_ids)]))
    num_events = len(starts)

    prec_sum = {k: 0.0 for k in ks}
    rec_sum = {k: 0.0 for k in ks}
    ndcg_sum = {k: 0.0 for k in ks}

    for s, e in zip(starts, ends):
        pos_movie = int(pos_movie_indices[s])
        cand_movies = movie_indices_for_scores[s:e]
        cand_scores = scores_for_scores[s:e]

        # Find the positive position within this group (it should exist exactly once).
        # Determine rank using the same tie-breaking style we used in baselines:
        # sort by score desc, then by movie_index desc (so larger movie_index comes first).
        # We implement rank by computing the ordered index.
        cand_movies_i32 = cand_movies.astype(np.int32, copy=False)
        cand_scores_f32 = cand_scores.astype(np.float32, copy=False)

        order = np.lexsort((-cand_movies_i32, -cand_scores_f32))
        pos_pos = int(np.where(cand_movies_i32[order] == pos_movie)[0][0])
        rank_pos = pos_pos + 1  # 1-indexed rank

        for k in ks:
            if rank_pos <= k:
                prec_sum[k] += 1.0 / float(k)
                rec_sum[k] += 1.0
                ndcg_sum[k] += 1.0 / float(np.log2(rank_pos + 1.0))

    return {
        **{f"precision@{k}": prec_sum[k] / num_events for k in ks},
        **{f"recall@{k}": rec_sum[k] / num_events for k in ks},
        **{f"ndcg@{k}": ndcg_sum[k] / num_events for k in ks},
    }


def _build_user_item_train(train_df: pd.DataFrame, num_users: int, num_items: int) -> sparse.csr_matrix:
    # user_index, movie_index, rating
    u = train_df["user_index"].to_numpy(dtype=np.int32, copy=False)
    i = train_df["movie_index"].to_numpy(dtype=np.int32, copy=False)
    r = train_df["rating"].to_numpy(dtype=np.float32, copy=False)
    mat = sparse.csr_matrix((r, (u, i)), shape=(num_users, num_items))
    return mat


def train_item_item_knn(
    train_df: pd.DataFrame,
    num_users: int,
    num_items: int,
    *,
    neighbor_k: int = 50,
    sim_eps: float = 1e-12,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Item-item cosine similarity using:
    sim(i,j) = <r_i, r_j> / (||r_i|| ||r_j||),
    where r_i is the item-rating vector across users.

    Returns:
      neighbors[i] = array of neighbor item indices (size <= neighbor_k)
      sims[i]      = corresponding similarity values
    """
    R = _build_user_item_train(train_df, num_users, num_items)  # CSR
    # Normalize columns (items) by their L2 norm.
    # Need CSC for efficient column operations.
    R_csc = R.tocsc(copy=False)
    col_sq_sum = np.asarray(R_csc.power(2).sum(axis=0)).reshape(-1)  # (num_items,)
    col_norm = np.sqrt(col_sq_sum) + sim_eps

    # Construct normalized sparse matrix by scaling columns:
    # normalized_value(u,i) = r_ui / ||r_i||
    inv_norm = sparse.diags(1.0 / col_norm, format="csc")
    Rn = R_csc @ inv_norm  # still sparse

    # Item-item similarity sparse matrix:
    # S = Rn^T * Rn  (items x items)
    S = (Rn.T @ Rn).tocsr()
    S.setdiag(0.0)

    neighbors: list[np.ndarray] = []
    sims: list[np.ndarray] = []

    for item_i in range(num_items):
        row_start = S.indptr[item_i]
        row_end = S.indptr[item_i + 1]
        idx = S.indices[row_start:row_end]
        vals = S.data[row_start:row_end]
        if vals.size == 0:
            neighbors.append(np.empty((0,), dtype=np.int32))
            sims.append(np.empty((0,), dtype=np.float32))
            continue

        # Pick top neighbors by similarity (descending).
        if vals.size > neighbor_k:
            top = np.argpartition(-vals, neighbor_k - 1)[:neighbor_k]
            idx = idx[top]
            vals = vals[top]
        order = np.argsort(-vals)
        idx = idx[order].astype(np.int32, copy=False)
        vals = vals[order].astype(np.float32, copy=False)

        neighbors.append(idx)
        sims.append(vals)

    return neighbors, sims


def score_knn_user_items(
    user_ratings: dict[int, float],
    candidate_items: np.ndarray,
    neighbors: list[np.ndarray],
    sims: list[np.ndarray],
) -> np.ndarray:
    """
    candidate score for item i:
      score(u,i) = sum_{j in N(i) intersect Rated(u)} sim(i,j) * r(u,j)
    """
    out = np.zeros(len(candidate_items), dtype=np.float32)
    for k, item_i in enumerate(candidate_items):
        neigh = neighbors[int(item_i)]
        if neigh.size == 0:
            out[k] = 0.0
            continue
        svals = sims[int(item_i)]
        total = 0.0
        for nb, sv in zip(neigh, svals):
            nb_int = int(nb)
            r = user_ratings.get(nb_int)
            if r is not None:
                total += float(sv) * float(r)
        out[k] = total
    return out


def score_knn_user_item_normalized(
    user_ratings: dict[int, float],
    item_i: int,
    neighbors: list[np.ndarray],
    sims: list[np.ndarray],
    *,
    user_mean: float,
    eps: float = 1e-12,
) -> float:
    """
    Rating predictor for KNN:
      pred(u,i) = sum_j sim(i,j) * r(u,j) / sum_j |sim(i,j)|
    Falls back to user's mean if no neighbors contribute.
    """
    neigh = neighbors[int(item_i)]
    if neigh.size == 0:
        return float(user_mean)
    svals = sims[int(item_i)]
    total = 0.0
    denom = 0.0
    for nb, sv in zip(neigh, svals):
        nb_int = int(nb)
        r = user_ratings.get(nb_int)
        if r is None:
            continue
        total += float(sv) * float(r)
        denom += abs(float(sv))
    if denom <= eps:
        return float(user_mean)
    return float(total / denom)


def evaluate_knn_ranking_for_test_negs(
    test_negs_df: pd.DataFrame,
    neighbors: list[np.ndarray],
    sims: list[np.ndarray],
    train_df_for_user_history: pd.DataFrame,
    *,
    ks: list[int],
) -> dict[str, float]:
    # Build per-user ratings dict from training history.
    user_hist: dict[int, dict[int, float]] = {}
    # (Keep only train_df_for_user_history, not val/test.)
    for u, i, r in zip(
        train_df_for_user_history["user_index"].to_numpy(dtype=np.int32, copy=False),
        train_df_for_user_history["movie_index"].to_numpy(dtype=np.int32, copy=False),
        train_df_for_user_history["rating"].to_numpy(dtype=np.float32, copy=False),
    ):
        user_hist.setdefault(int(u), {})[int(i)] = float(r)

    test_negs_df_sorted = test_negs_df.sort_values(["event_id", "neg_movie_index"], kind="stable")
    grouped = test_negs_df_sorted.groupby("event_id", sort=False)

    event_ids = []
    pos_movie_indices = []
    cand_movie_indices = []
    cand_scores = []

    # We'll build candidate scoring arrays and then compute metrics from scores.
    for event_id, g in grouped:
        event_id = int(event_id)
        u = int(g["user_index"].iloc[0])
        pos_movie = int(g["pos_movie_index"].iloc[0])
        negs = g["neg_movie_index"].to_numpy(dtype=np.int32, copy=False)
        candidates = np.concatenate(([pos_movie], negs))
        scores = score_knn_user_items(
            user_ratings=user_hist.get(u, {}),
            candidate_items=candidates,
            neighbors=neighbors,
            sims=sims,
        )

        event_ids.extend([event_id] * len(candidates))
        pos_movie_indices.extend([pos_movie] * len(candidates))
        cand_movie_indices.extend(candidates.tolist())
        cand_scores.extend(scores.tolist())

    event_ids = np.asarray(event_ids, dtype=np.int64)
    pos_movie_indices = np.asarray(pos_movie_indices, dtype=np.int32)
    cand_movie_indices = np.asarray(cand_movie_indices, dtype=np.int32)
    cand_scores = np.asarray(cand_scores, dtype=np.float32)

    return _metrics_from_scores_onepos(
        event_ids=event_ids,
        pos_movie_indices=pos_movie_indices,
        movie_indices_for_scores=cand_movie_indices,
        scores_for_scores=cand_scores,
        ks=ks,
    )


def train_and_evaluate_spark_als_residual_biases(
    *,
    spark,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_events_df: pd.DataFrame,
    val_negs_df: pd.DataFrame,
    test_events_df: pd.DataFrame,
    test_negs_df: pd.DataFrame,
    num_users: int,
    num_items: int,
    rank: int,
    reg_param: float,
    max_iter: int,
    bias_lambda: float,
    num_user_blocks: int,
    num_item_blocks: int,
    seed: int,
    ks: list[int],
    export_artifacts_dir: pathlib.Path | None = None,
    compute_val_metrics: bool = True,
    compute_test_metrics: bool = True,
) -> dict:
    """
    Train Spark ALS on residual ratings:
      r_resid = rating - mu - b_u - b_i
    Then prediction score:
      r_pred = mu + b_u + b_i + ALS(u,i)

    We tune hyperparameters on val, evaluated by NDCG@10 computed using val_negs.
    """
    from pyspark.ml.recommendation import ALS
    from pyspark.sql import functions as F

    # Biases from training data (regularized means).
    u = train_df["user_index"].to_numpy(dtype=np.int32, copy=False)
    i = train_df["movie_index"].to_numpy(dtype=np.int32, copy=False)
    r = train_df["rating"].to_numpy(dtype=np.float32, copy=False)
    mu = float(np.mean(r, dtype=np.float64))

    count_u = np.bincount(u, minlength=num_users).astype(np.float32)
    count_i = np.bincount(i, minlength=num_items).astype(np.float32)

    b_u = np.zeros(num_users, dtype=np.float32)
    b_i = np.zeros(num_items, dtype=np.float32)

    # Initialize b_u, b_i with one round of residual means.
    res = r - mu
    sum_u = np.bincount(u, weights=res, minlength=num_users).astype(np.float32)
    sum_i = np.bincount(i, weights=res, minlength=num_items).astype(np.float32)
    b_u = sum_u / (bias_lambda + count_u)
    b_i = sum_i / (bias_lambda + count_i)

    # Helper: convert numpy arrays to spark tables for joins.
    bu_sdf = spark.createDataFrame(pd.DataFrame({"user_index": np.arange(num_users, dtype=np.int32), "b_u": b_u}))
    bi_sdf = spark.createDataFrame(pd.DataFrame({"movie_index": np.arange(num_items, dtype=np.int32), "b_i": b_i}))

    def _to_spark(df_pd: pd.DataFrame, name: str):
        return spark.createDataFrame(df_pd[["user_index", "movie_index", "rating"]].copy())

    train_sdf = _to_spark(train_df, "train")

    bu_bi_train = train_sdf.join(bu_sdf, on="user_index", how="left").join(bi_sdf, on="movie_index", how="left")
    # residual column
    bu_bi_train = bu_bi_train.withColumn("rating_resid", bu_bi_train["rating"] - float(mu) - bu_bi_train["b_u"] - bu_bi_train["b_i"])
    # Keep ALS-friendly columns
    als_train = bu_bi_train.select("user_index", "movie_index", "rating_resid")

    # Train ALS for residual
    als = ALS(
        userCol="user_index",
        itemCol="movie_index",
        ratingCol="rating_resid",
        rank=rank,
        maxIter=max_iter,
        regParam=float(reg_param),
        numUserBlocks=int(num_user_blocks),
        numItemBlocks=int(num_item_blocks),
        implicitPrefs=False,
        nonnegative=False,
        seed=seed,
        # Use NaN predictions for cold-start so candidate sets stay intact.
        # We'll convert NaN residual predictions to 0 downstream.
        coldStartStrategy="nan",
    )
    als_model = als.fit(als_train)

    # Optionally export ALS latent factors + biases for use in a CLI demo.
    if export_artifacts_dir is not None:
        export_artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Save scalar/bias parameters.
        np.save(export_artifacts_dir / "mu.npy", np.asarray(mu, dtype=np.float32))
        np.save(export_artifacts_dir / "bias_bu.npy", b_u.astype(np.float32, copy=False))
        np.save(export_artifacts_dir / "bias_bi.npy", b_i.astype(np.float32, copy=False))

        # Export user/item factors.
        user_factors = np.zeros((num_users, int(rank)), dtype=np.float32)
        item_factors = np.zeros((num_items, int(rank)), dtype=np.float32)

        # Spark ALS factor ids are contiguous starting at 0.
        # We fill missing ids (e.g. cold-start) with zeros.
        for row in als_model.userFactors.select("id", "features").collect():
            user_factors[int(row["id"])] = np.asarray(row["features"], dtype=np.float32)

        for row in als_model.itemFactors.select("id", "features").collect():
            item_factors[int(row["id"])] = np.asarray(row["features"], dtype=np.float32)

        np.save(export_artifacts_dir / "user_factors.npy", user_factors)
        np.save(export_artifacts_dir / "item_factors.npy", item_factors)

        meta = {
            "rank": int(rank),
            "reg_param": float(reg_param),
            "max_iter": int(max_iter),
            "bias_lambda": float(bias_lambda),
            "num_users": int(num_users),
            "num_items": int(num_items),
            "seed": int(seed),
        }
        with open(export_artifacts_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    # Function to predict for arbitrary (u,i) pairs and then add biases back.
    def _predict_pairs(pairs_df_pd: pd.DataFrame, pairs_name: str) -> pd.DataFrame:
        sdf = spark.createDataFrame(pairs_df_pd[["user_index", "movie_index"]].copy())
        pred = als_model.transform(sdf).select("user_index", "movie_index", "prediction")
        # Join biases and add mu.
        pred = pred.join(bu_sdf, on="user_index", how="left").join(bi_sdf, on="movie_index", how="left")
        # Spark may output NaN for cold-start. Treat residual as 0 in that case.
        residual_pred = F.when(F.isnan(pred["prediction"]), F.lit(0.0)).otherwise(pred["prediction"])
        pred = pred.withColumn("rating_pred", residual_pred + float(mu) + pred["b_u"] + pred["b_i"])
        # Return as pandas for evaluation.
        return pred.select("user_index", "movie_index", "rating_pred").toPandas()

    # Rating prediction metrics on val/test.
    if compute_val_metrics:
        val_pred_pairs = val_df[["user_index", "movie_index"]].copy()
        val_pred = _predict_pairs(val_pred_pairs, "val_pred")
        val_pred = val_pred.merge(val_df[["user_index", "movie_index", "rating"]], on=["user_index", "movie_index"], how="inner")
        rmse_val, mae_val = _rmse_mae(
            val_pred["rating"].to_numpy(dtype=np.float32),
            val_pred["rating_pred"].to_numpy(dtype=np.float32),
        )
    else:
        rmse_val, mae_val = None, None

    if compute_test_metrics:
        test_pred_pairs = test_df[["user_index", "movie_index"]].copy()
        test_pred = _predict_pairs(test_pred_pairs, "test_pred")
        test_pred = test_pred.merge(test_df[["user_index", "movie_index", "rating"]], on=["user_index", "movie_index"], how="inner")
        rmse_test, mae_test = _rmse_mae(
            test_pred["rating"].to_numpy(dtype=np.float32),
            test_pred["rating_pred"].to_numpy(dtype=np.float32),
        )
    else:
        rmse_test, mae_test = None, None

    # Ranking evaluation on val/test candidate sets.
    # IMPORTANT: Do this fully in Spark to avoid driver OOM from collecting millions of predictions.
    from pyspark.sql.window import Window

    def _ranking_metrics_spark(events_df_pd: pd.DataFrame, negs_df_pd: pd.DataFrame) -> dict[str, float]:
        """
        One-positive-per-event candidate construction:
          cand = {pos(event)} U {negs(event)} with is_pos flag.
        Metrics are computed as mean over held-out events of:
          precision@k = (1/k) * I(rank_pos<=k)
          recall@k    = I(rank_pos<=k)
          ndcg@k      = I(rank_pos<=k) / log2(rank_pos+1)
        """
        pos_events = events_df_pd[["event_id", "user_index", "pos_movie_index"]].copy()
        pos_tmp = pos_events.rename(columns={"pos_movie_index": "movie_index"})
        pos_sdf = spark.createDataFrame(pos_tmp[["event_id", "user_index", "movie_index"]].copy())
        pos_sdf = pos_sdf.withColumn("is_pos", F.lit(1))

        neg_tmp = negs_df_pd[["event_id", "user_index", "neg_movie_index"]].copy()
        neg_tmp = neg_tmp.rename(columns={"neg_movie_index": "movie_index"})
        neg_sdf = spark.createDataFrame(neg_tmp[["event_id", "user_index", "movie_index"]].copy())
        neg_sdf = neg_sdf.withColumn("is_pos", F.lit(0))

        cand_sdf = pos_sdf.unionByName(neg_sdf)

        # Predict residuals then add biases back to get rating_pred.
        pred = als_model.transform(cand_sdf.select("event_id", "user_index", "movie_index")).select(
            "event_id",
            "user_index",
            "movie_index",
            "prediction",
        )
        pred = pred.join(bu_sdf, on="user_index", how="left").join(bi_sdf, on="movie_index", how="left")
        residual_pred = F.when(F.isnan(pred["prediction"]), F.lit(0.0)).otherwise(pred["prediction"])
        pred = pred.withColumn("rating_pred", residual_pred + float(mu) + pred["b_u"] + pred["b_i"])

        cand_pred = pred.join(cand_sdf.select("event_id", "user_index", "movie_index", "is_pos"), on=["event_id", "user_index", "movie_index"], how="inner")

        w = Window.partitionBy("event_id").orderBy(F.col("rating_pred").desc(), F.col("movie_index").desc())
        ranked = cand_pred.withColumn("rn", F.row_number().over(w))

        pos_ranks = ranked.filter(F.col("is_pos") == 1).select("event_id", "rn")

        # Precompute log2 base (Spark doesn't guarantee log2 across all versions).
        log2_denom = F.log(F.col("rn").cast("double") + F.lit(1.0)) / F.log(F.lit(2.0))
        ndcg_term = F.when(F.col("rn") <= F.lit(9999999), F.lit(1.0) / log2_denom).otherwise(F.lit(0.0))

        agg_exprs = []
        for k in ks:
            within = F.col("rn") <= F.lit(int(k))
            # precision@k = (1/k) when within else 0
            agg_exprs.append(F.avg(F.when(within, F.lit(1.0 / float(k))).otherwise(F.lit(0.0))).alias(f"precision@{k}"))
            # recall@k = 1 when within else 0
            agg_exprs.append(F.avg(F.when(within, F.lit(1.0)).otherwise(F.lit(0.0))).alias(f"recall@{k}"))
            # ndcg@k = 1/log2(rn+1) when within else 0
            agg_exprs.append(
                F.avg(F.when(within, F.lit(1.0) / log2_denom).otherwise(F.lit(0.0))).alias(f"ndcg@{k}")
            )

        row = pos_ranks.agg(*agg_exprs).collect()[0].asDict()
        # Ensure native floats
        return {k: float(v) for k, v in row.items()}

    if compute_val_metrics:
        val_rank_metrics = _ranking_metrics_spark(val_events_df, val_negs_df)
        ndcg_at_k_10 = float(val_rank_metrics.get("ndcg@10", 0.0))
    else:
        val_rank_metrics = None
        ndcg_at_k_10 = None

    if compute_test_metrics:
        test_rank_metrics = _ranking_metrics_spark(test_events_df, test_negs_df)
    else:
        test_rank_metrics = None

    return {
        "rank": rank,
        "reg_param": reg_param,
        "max_iter": max_iter,
        "bias_lambda": bias_lambda,
        "val_rating": None if rmse_val is None else {"rmse": rmse_val, "mae": mae_val},
        "test_rating": None if rmse_test is None else {"rmse": rmse_test, "mae": mae_test},
        "val_ranking": val_rank_metrics,
        "test_ranking": test_rank_metrics,
        "val_ndcg@10": ndcg_at_k_10,
        "mu": mu,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collaborative filtering models (KNN + ALS).")
    parser.add_argument("--dataset", choices=["1m", "10m", "both"], default="1m")
    parser.add_argument("--ks", type=str, default="5,10,20")

    # KNN
    parser.add_argument("--knn-neighbors", type=str, default="20,50")

    # ALS
    parser.add_argument("--als-ranks", type=str, default="10,20")
    parser.add_argument("--als-regs", type=str, default="0.1,1.0")
    parser.add_argument("--als-max-iter", type=int, default=10)
    parser.add_argument("--als-bias-lambda", type=float, default=25.0)
    parser.add_argument(
        "--als-user-blocks",
        type=int,
        default=None,
        help="ALS user block count. Default: 10 for 1m, 40 for 10m.",
    )
    parser.add_argument(
        "--als-item-blocks",
        type=int,
        default=None,
        help="ALS item block count. Default: 10 for 1m, 40 for 10m.",
    )
    parser.add_argument("--spark-master", type=str, default="local[4]")
    parser.add_argument("--spark-driver-memory", type=str, default="4g")
    parser.add_argument("--spark-executor-memory", type=str, default="4g")
    parser.add_argument("--spark-default-parallelism", type=int, default=4)
    parser.add_argument("--spark-shuffle-partitions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-negatives", type=int, default=99)
    parser.add_argument(
        "--max-ranking-events",
        type=int,
        default=None,
        help="Maximum held-out events to evaluate for ranking. Default: 20000 for both 1m and 10m.",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="knn_tune,knn_final,als_tune,als_final",
        help="Comma-separated stages: knn_tune,knn_final,als_tune,als_final",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from saved progress.json and skip completed stages.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run requested stages even if marked completed in progress.json.",
    )

    args = parser.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x]
    knn_candidates = [int(x) for x in args.knn_neighbors.split(",") if x]
    als_ranks = [int(x) for x in args.als_ranks.split(",") if x]
    als_regs = [float(x) for x in args.als_regs.split(",") if x]
    requested_stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    valid_stages = {"knn_tune", "knn_final", "als_tune", "als_final"}
    bad_stages = [s for s in requested_stages if s not in valid_stages]
    if bad_stages:
        raise SystemExit(f"Unknown stage(s): {bad_stages}. Valid: {sorted(valid_stages)}")

    # Spark session
    from pyspark.sql import SparkSession
    spark = (
        SparkSession.builder.master(args.spark_master)
        .appName("collab_filtering_step3")
        .config("spark.driver.memory", args.spark_driver_memory)
        .config("spark.executor.memory", args.spark_executor_memory)
        .config("spark.default.parallelism", str(args.spark_default_parallelism))
        .config("spark.sql.shuffle.partitions", str(args.spark_shuffle_partitions))
        .getOrCreate()
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    targets = ["1m", "10m"] if args.dataset == "both" else [args.dataset]
    for key in targets:
        als_user_blocks = args.als_user_blocks if args.als_user_blocks is not None else (10 if key == "1m" else 40)
        als_item_blocks = args.als_item_blocks if args.als_item_blocks is not None else (10 if key == "1m" else 40)
        processed_dir = PROCESSED_DIR / key
        meta = json.load(open(processed_dir / "meta.json", "r"))
        num_users = int(meta["num_users"])
        num_items = int(meta["num_movies"])
        out_dir = RESULTS_DIR / key
        out_dir.mkdir(parents=True, exist_ok=True)
        progress_path = out_dir / "progress.json"

        progress = {
            "dataset": key,
            "completed_stages": [],
            "knn_best": None,
            "knn_results": None,
            "als_best": None,
            "als_results": None,
        }
        if args.resume and progress_path.exists():
            progress = json.load(open(progress_path, "r"))
            print(f"[resume] {key}: loaded completed_stages={progress.get('completed_stages', [])}")

        def save_progress() -> None:
            with open(progress_path, "w") as f:
                json.dump(progress, f, indent=2)

        print(f"[prep] {key}: loading Step-1 artifacts...")
        train_df = read_csv_gz(
            processed_dir / "train.csv.gz",
            dtypes={"user_index": np.int32, "movie_index": np.int32, "rating": np.float32, "timestamp": np.int64},
        )
        val_df = read_csv_gz(
            processed_dir / "val.csv.gz",
            dtypes={"user_index": np.int32, "movie_index": np.int32, "rating": np.float32, "timestamp": np.int64},
        )
        test_df = read_csv_gz(
            processed_dir / "test.csv.gz",
            dtypes={"user_index": np.int32, "movie_index": np.int32, "rating": np.float32, "timestamp": np.int64},
        )
        ratings_clean_df = read_csv_gz(
            processed_dir / "ratings_clean.csv.gz",
            dtypes={"user_index": np.int32, "movie_index": np.int32, "rating": np.float32, "timestamp": np.int64},
        )
        max_ranking_events = default_max_ranking_events(key) if args.max_ranking_events is None else args.max_ranking_events
        val_events_df, val_negs_df, val_eval_summary = build_candidate_groups(
            split_df=val_df,
            rated_source_df=ratings_clean_df,
            num_movies=num_items,
            num_negatives=args.num_negatives,
            seed=args.seed + 300,
            max_events=max_ranking_events,
        )
        test_events_df, test_negs_df, test_eval_summary = build_candidate_groups(
            split_df=test_df,
            rated_source_df=ratings_clean_df,
            num_movies=num_items,
            num_negatives=args.num_negatives,
            seed=args.seed + 400,
            max_events=max_ranking_events,
        )

        def should_run(stage: str) -> bool:
            if stage not in requested_stages:
                return False
            if args.resume and (not args.force) and stage in progress.get("completed_stages", []):
                print(f"[skip] {key}: stage {stage} already completed in progress.json")
                return False
            return True

        if should_run("knn_tune"):
            print(f"[knn] {key}: tuning item-item KNN...")
            knn_best = None
            for neigh_k in knn_candidates:
                neighbors, sims = train_item_item_knn(train_df, num_users, num_items, neighbor_k=neigh_k)
                knn_val_metrics = evaluate_knn_ranking_for_test_negs(
                    test_negs_df=val_negs_df,
                    neighbors=neighbors,
                    sims=sims,
                    train_df_for_user_history=train_df,
                    ks=ks,
                )
                ndcg10 = float(knn_val_metrics.get("ndcg@10", 0.0))
                print(f"[knn] {key}: neigh_k={neigh_k} val_ndcg@10={ndcg10:.6f}")
                if knn_best is None or ndcg10 > knn_best["ndcg10"]:
                    knn_best = {"neigh_k": neigh_k, "ndcg10": ndcg10, "val_metrics": knn_val_metrics}
            progress["knn_best"] = knn_best
            progress["completed_stages"] = sorted(set(progress["completed_stages"] + ["knn_tune"]))
            save_progress()

        if should_run("knn_final"):
            if progress.get("knn_best") is None:
                raise SystemExit("knn_final requires knn_tune result in progress.json")
            knn_best = progress["knn_best"]
            print(f"[knn] {key}: training final KNN with neigh_k={knn_best['neigh_k']}...")
            trainval_df = pd.concat([train_df, val_df], ignore_index=True)
            neighbors_final, sims_final = train_item_item_knn(
                trainval_df, num_users, num_items, neighbor_k=int(knn_best["neigh_k"])
            )
            knn_test_metrics = evaluate_knn_ranking_for_test_negs(
                test_negs_df=test_negs_df,
                neighbors=neighbors_final,
                sims=sims_final,
                train_df_for_user_history=trainval_df,
                ks=ks,
            )

            user_hist = {}
            for u0, i0, r0 in zip(
                trainval_df["user_index"].to_numpy(dtype=np.int32, copy=False),
                trainval_df["movie_index"].to_numpy(dtype=np.int32, copy=False),
                trainval_df["rating"].to_numpy(dtype=np.float32, copy=False),
            ):
                user_hist.setdefault(int(u0), {})[int(i0)] = float(r0)
            user_mean = {u0: float(np.mean(list(d.values()))) for u0, d in user_hist.items()}
            global_mean = float(trainval_df["rating"].mean())
            test_u = test_df["user_index"].to_numpy(dtype=np.int32, copy=False)
            test_i = test_df["movie_index"].to_numpy(dtype=np.int32, copy=False)
            test_r = test_df["rating"].to_numpy(dtype=np.float32, copy=False)
            pred = np.zeros_like(test_r, dtype=np.float32)
            for idx in range(len(test_r)):
                u0 = int(test_u[idx])
                i0 = int(test_i[idx])
                pred[idx] = score_knn_user_item_normalized(
                    user_ratings=user_hist.get(u0, {}),
                    item_i=i0,
                    neighbors=neighbors_final,
                    sims=sims_final,
                    user_mean=user_mean.get(u0, global_mean),
                )
            rmse_knn, mae_knn = _rmse_mae(test_r, pred)

            progress["knn_results"] = {
                "neighbor_k": int(knn_best["neigh_k"]),
                "tuning_val_ranking": knn_best["val_metrics"],
                "test_ranking": knn_test_metrics,
                "training_data": "trainval",
                "test_rating": {"rmse": rmse_knn, "mae": mae_knn},
                "ranking_eval": {"val": val_eval_summary, "test": test_eval_summary},
            }
            progress["completed_stages"] = sorted(set(progress["completed_stages"] + ["knn_final"]))
            save_progress()

        if should_run("als_tune"):
            print(f"[als] {key}: tuning ALS rank/reg on val...")
            als_best = None
            for rank in als_ranks:
                for reg in als_regs:
                    out = train_and_evaluate_spark_als_residual_biases(
                        spark=spark,
                        train_df=train_df,
                        val_df=val_df,
                        test_df=test_df,
                        val_events_df=val_events_df[["event_id", "user_index", "pos_movie_index"]],
                        val_negs_df=val_negs_df[["event_id", "user_index", "pos_movie_index", "neg_movie_index"]],
                        test_events_df=test_events_df[["event_id", "user_index", "pos_movie_index"]],
                        test_negs_df=test_negs_df[["event_id", "user_index", "pos_movie_index", "neg_movie_index"]],
                        num_users=num_users,
                        num_items=num_items,
                        rank=rank,
                        reg_param=reg,
                        max_iter=args.als_max_iter,
                        bias_lambda=args.als_bias_lambda,
                        num_user_blocks=als_user_blocks,
                        num_item_blocks=als_item_blocks,
                        seed=args.seed,
                        ks=ks,
                        compute_test_metrics=False,
                    )
                    ndcg10 = float(out.get("val_ndcg@10", 0.0))
                    print(f"[als] {key}: rank={rank} reg={reg} val_ndcg@10={ndcg10:.6f}")
                    if als_best is None or ndcg10 > als_best["val_ndcg10"]:
                        als_best = {
                            "rank": rank,
                            "reg": reg,
                            "val_ndcg10": ndcg10,
                            "val_ranking": out["val_ranking"],
                            "val_rating": out["val_rating"],
                        }
            progress["als_best"] = als_best
            progress["completed_stages"] = sorted(set(progress["completed_stages"] + ["als_tune"]))
            save_progress()

        if should_run("als_final"):
            if progress.get("als_best") is None:
                raise SystemExit("als_final requires als_tune result in progress.json")
            als_best = progress["als_best"]
            print(f"[als] {key}: training final ALS rank={als_best['rank']} reg={als_best['reg']}...")
            trainval_df = pd.concat([train_df, val_df], ignore_index=True)
            export_dir = MODELS_DIR / key / "als_residual"
            final_out = train_and_evaluate_spark_als_residual_biases(
                spark=spark,
                train_df=trainval_df,
                val_df=val_df,
                test_df=test_df,
                val_events_df=val_events_df[["event_id", "user_index", "pos_movie_index"]],
                val_negs_df=val_negs_df[["event_id", "user_index", "pos_movie_index", "neg_movie_index"]],
                test_events_df=test_events_df[["event_id", "user_index", "pos_movie_index"]],
                test_negs_df=test_negs_df[["event_id", "user_index", "pos_movie_index", "neg_movie_index"]],
                num_users=num_users,
                num_items=num_items,
                rank=int(als_best["rank"]),
                reg_param=float(als_best["reg"]),
                max_iter=args.als_max_iter,
                bias_lambda=args.als_bias_lambda,
                num_user_blocks=als_user_blocks,
                num_item_blocks=als_item_blocks,
                seed=args.seed,
                ks=ks,
                export_artifacts_dir=export_dir,
                compute_val_metrics=False,
                compute_test_metrics=True,
            )
            progress["als_results"] = {
                "best_rank": int(als_best["rank"]),
                "best_reg_param": float(als_best["reg"]),
                "training_data": "trainval",
                "tuning_val_ranking": als_best.get("val_ranking"),
                "tuning_val_rating": als_best.get("val_rating"),
                "test_ranking": final_out["test_ranking"],
                "test_rating": final_out["test_rating"],
                "ranking_eval": {"val": val_eval_summary, "test": test_eval_summary},
            }
            progress["completed_stages"] = sorted(set(progress["completed_stages"] + ["als_final"]))
            save_progress()

        results = {
            "dataset": key,
            "knn": progress.get("knn_results"),
            "als": progress.get("als_results"),
            "ks": ks,
            "completed_stages": progress.get("completed_stages", []),
        }
        with open(out_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"[done] {key}: wrote results to {out_dir / 'results.json'}")

    spark.stop()


if __name__ == "__main__":
    main()
