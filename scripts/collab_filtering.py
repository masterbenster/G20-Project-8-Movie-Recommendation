import argparse
import json
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = DATA_DIR / "results" / "collab_filtering"


def _read_csv_gz(path: pathlib.Path, dtypes=None, usecols=None) -> pd.DataFrame:
    if dtypes is None and usecols is None:
        return pd.read_csv(path, compression="gzip")
    if dtypes is None:
        return pd.read_csv(path, compression="gzip", usecols=usecols)
    return pd.read_csv(path, compression="gzip", usecols=usecols, dtype=dtypes)


def _rmse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(np.square(err))))
    mae = float(np.mean(np.abs(err)))
    return rmse, mae


def _metrics_from_scores_onepos(
    user_indices: np.ndarray,
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
    - For each user u, there are M candidate movies (pos + negatives) and we have:
      user_indices length = num_users * M
      movie_indices_for_scores includes the positive movie exactly once per user.
    - We assume score arrays correspond 1:1 with movie_indices_for_scores.
    """
    ks = sorted(ks)
    max_k = max(ks)

    # Ensure stable grouping by user.
    order_u = np.argsort(user_indices, kind="stable")
    user_indices = user_indices[order_u]
    pos_movie_indices = pos_movie_indices[order_u]
    movie_indices_for_scores = movie_indices_for_scores[order_u]
    scores_for_scores = scores_for_scores[order_u]

    # Group boundaries
    boundaries = np.flatnonzero(user_indices[1:] != user_indices[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(user_indices)]))
    num_users = len(starts)

    prec_sum = {k: 0.0 for k in ks}
    rec_sum = {k: 0.0 for k in ks}
    ndcg_sum = {k: 0.0 for k in ks}

    for s, e in zip(starts, ends):
        u = int(user_indices[s])
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
        **{f"precision@{k}": prec_sum[k] / num_users for k in ks},
        **{f"recall@{k}": rec_sum[k] / num_users for k in ks},
        **{f"ndcg@{k}": ndcg_sum[k] / num_users for k in ks},
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
    test_df: pd.DataFrame,
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

    # Candidate set per user: [pos] + negs (derived from test_negs_df)
    # test_df has one row per user (pos)
    user_pos = test_df[["user_index", "movie_index"]].copy()
    user_pos = user_pos.rename(columns={"movie_index": "pos_movie_index"})

    # Sort for deterministic grouping
    test_negs_df_sorted = test_negs_df.sort_values(["user_index", "neg_movie_index"], kind="stable")
    grouped = test_negs_df_sorted.groupby("user_index", sort=False)

    user_indices = []
    pos_movie_indices = []
    cand_movie_indices = []
    cand_scores = []

    # We'll build candidate scoring arrays and then compute metrics from scores.
    max_k = max(ks)
    for u, g in grouped:
        u = int(u)
        pos_movie = int(user_pos.loc[user_pos["user_index"] == u, "pos_movie_index"].iloc[0])
        negs = g["neg_movie_index"].to_numpy(dtype=np.int32, copy=False)
        candidates = np.concatenate(([pos_movie], negs))
        scores = score_knn_user_items(
            user_ratings=user_hist.get(u, {}),
            candidate_items=candidates,
            neighbors=neighbors,
            sims=sims,
        )

        user_indices.extend([u] * len(candidates))
        pos_movie_indices.extend([pos_movie] * len(candidates))
        cand_movie_indices.extend(candidates.tolist())
        cand_scores.extend(scores.tolist())

    user_indices = np.asarray(user_indices, dtype=np.int32)
    pos_movie_indices = np.asarray(pos_movie_indices, dtype=np.int32)
    cand_movie_indices = np.asarray(cand_movie_indices, dtype=np.int32)
    cand_scores = np.asarray(cand_scores, dtype=np.float32)

    return _metrics_from_scores_onepos(
        user_indices=user_indices,
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
    val_negs_df: pd.DataFrame,
    test_negs_df: pd.DataFrame,
    num_users: int,
    num_items: int,
    rank: int,
    reg_param: float,
    max_iter: int,
    bias_lambda: float,
    seed: int,
    ks: list[int],
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
    val_sdf = _to_spark(val_df, "val")
    test_sdf = _to_spark(test_df, "test")

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
        implicitPrefs=False,
        nonnegative=False,
        seed=seed,
        # Use NaN predictions for cold-start so candidate sets stay intact.
        # We'll convert NaN residual predictions to 0 downstream.
        coldStartStrategy="nan",
    )
    als_model = als.fit(als_train)

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
    val_pred_pairs = val_df[["user_index", "movie_index"]].copy()
    val_pred = _predict_pairs(val_pred_pairs, "val_pred")
    val_pred = val_pred.merge(val_df[["user_index", "movie_index", "rating"]], on=["user_index", "movie_index"], how="inner")
    rmse_val, mae_val = _rmse_mae(val_pred["rating"].to_numpy(dtype=np.float32), val_pred["rating_pred"].to_numpy(dtype=np.float32))

    test_pred_pairs = test_df[["user_index", "movie_index"]].copy()
    test_pred = _predict_pairs(test_pred_pairs, "test_pred")
    test_pred = test_pred.merge(test_df[["user_index", "movie_index", "rating"]], on=["user_index", "movie_index"], how="inner")
    rmse_test, mae_test = _rmse_mae(test_pred["rating"].to_numpy(dtype=np.float32), test_pred["rating_pred"].to_numpy(dtype=np.float32))

    # Ranking evaluation on val_negs and test_negs using candidate sets:
    def _build_candidates_and_score(negs_df: pd.DataFrame, pos_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        # negs_df columns: user_index, neg_movie_index
        negs_pairs = negs_df[["user_index", "neg_movie_index"]].copy().rename(columns={"neg_movie_index": "movie_index"})
        pos_pairs = pos_df[["user_index", "movie_index"]].copy()
        pos_pairs = pos_pairs.rename(columns={"movie_index": "movie_index"})

        # Create combined pairs for scoring.
        cand_pairs = pd.concat(
            [
                pos_pairs.assign(is_pos=1),
                negs_pairs.assign(is_pos=0),
            ],
            ignore_index=True,
        )
        pred = _predict_pairs(cand_pairs[["user_index", "movie_index"]], "cand_pred")
        pred = pred.merge(cand_pairs[["user_index", "movie_index", "is_pos"]], on=["user_index", "movie_index"], how="inner")
        return pred

    # Val candidates
    val_pos_df = val_df[["user_index", "movie_index"]].copy().rename(columns={"movie_index": "movie_index"})
    val_negs_pairs = val_negs_df.copy()
    val_pred = _build_candidates_and_score(val_negs_pairs.rename(columns={"neg_movie_index": "neg_movie_index"}), val_pos_df)
    # Compute NDCG@K with a numpy-based "one positive per user" construction.
    # We need exactly one positive candidate per user in the candidate set.
    # For each user, we create arrays for all scored items (pos + negs).
    # Get pos_movie_index per user from val_pos_df.
    pos_per_user = val_pos_df.set_index("user_index")["movie_index"].to_dict()

    user_list = []
    pos_movie_list = []
    movie_list = []
    score_list = []
    for u0, grp in val_pred.groupby("user_index"):
        u0 = int(u0)
        pos_movie = int(pos_per_user[u0])
        cand_movies = grp["movie_index"].to_numpy(dtype=np.int32, copy=False)
        cand_scores = grp["rating_pred"].to_numpy(dtype=np.float32, copy=False)
        # Ensure pos included once
        user_list.extend([u0] * len(cand_movies))
        pos_movie_list.extend([pos_movie] * len(cand_movies))
        movie_list.extend(cand_movies.tolist())
        score_list.extend(cand_scores.tolist())

    val_rank_metrics = _metrics_from_scores_onepos(
        user_indices=np.asarray(user_list, dtype=np.int32),
        pos_movie_indices=np.asarray(pos_movie_list, dtype=np.int32),
        movie_indices_for_scores=np.asarray(movie_list, dtype=np.int32),
        scores_for_scores=np.asarray(score_list, dtype=np.float32),
        ks=ks,
    )
    ndcg_at_k_10 = float(val_rank_metrics.get("ndcg@10", 0.0))

    # Test candidates
    test_pos_df = test_df[["user_index", "movie_index"]].copy().rename(columns={"movie_index": "movie_index"})
    test_pred_pairs_df = _build_candidates_and_score(test_negs_df.rename(columns={"neg_movie_index": "neg_movie_index"}), test_pos_df)
    pos_per_user_test = test_pos_df.set_index("user_index")["movie_index"].to_dict()
    user_list = []
    pos_movie_list = []
    movie_list = []
    score_list = []
    for u0, grp in test_pred_pairs_df.groupby("user_index"):
        u0 = int(u0)
        pos_movie = int(pos_per_user_test[u0])
        cand_movies = grp["movie_index"].to_numpy(dtype=np.int32, copy=False)
        cand_scores = grp["rating_pred"].to_numpy(dtype=np.float32, copy=False)
        user_list.extend([u0] * len(cand_movies))
        pos_movie_list.extend([pos_movie] * len(cand_movies))
        movie_list.extend(cand_movies.tolist())
        score_list.extend(cand_scores.tolist())
    test_rank_metrics = _metrics_from_scores_onepos(
        user_indices=np.asarray(user_list, dtype=np.int32),
        pos_movie_indices=np.asarray(pos_movie_list, dtype=np.int32),
        movie_indices_for_scores=np.asarray(movie_list, dtype=np.int32),
        scores_for_scores=np.asarray(score_list, dtype=np.float32),
        ks=ks,
    )

    return {
        "rank": rank,
        "reg_param": reg_param,
        "max_iter": max_iter,
        "bias_lambda": bias_lambda,
        "val_rating": {"rmse": rmse_val, "mae": mae_val},
        "test_rating": {"rmse": rmse_test, "mae": mae_test},
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
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x]
    knn_candidates = [int(x) for x in args.knn_neighbors.split(",") if x]
    als_ranks = [int(x) for x in args.als_ranks.split(",") if x]
    als_regs = [float(x) for x in args.als_regs.split(",") if x]

    # Spark session
    from pyspark.sql import SparkSession
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("collab_filtering_step3")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    targets = ["1m", "10m"] if args.dataset == "both" else [args.dataset]
    for key in targets:
        processed_dir = PROCESSED_DIR / key
        meta = json.load(open(processed_dir / "meta.json", "r"))
        num_users = int(meta["num_users"])
        num_items = int(meta["num_movies"])

        print(f"[prep] {key}: loading Step-1 artifacts...")
        train_df = _read_csv_gz(
            processed_dir / "train.csv.gz",
            dtypes={"user_index": np.int32, "movie_index": np.int32, "rating": np.float32, "timestamp": np.int64},
        )
        val_df = _read_csv_gz(
            processed_dir / "val.csv.gz",
            dtypes={"user_index": np.int32, "movie_index": np.int32, "rating": np.float32, "timestamp": np.int64},
        )
        test_df = _read_csv_gz(
            processed_dir / "test.csv.gz",
            dtypes={"user_index": np.int32, "movie_index": np.int32, "rating": np.float32, "timestamp": np.int64},
        )
        val_negs_df = _read_csv_gz(
            processed_dir / "val_negs.csv.gz",
            dtypes={"user_index": np.int32, "pos_movie_index": np.int32, "neg_movie_index": np.int32},
        )
        test_negs_df = _read_csv_gz(
            processed_dir / "test_negs.csv.gz",
            dtypes={"user_index": np.int32, "pos_movie_index": np.int32, "neg_movie_index": np.int32},
        )

        # ===== KNN tuning on val ranking (NDCG@10) =====
        print(f"[knn] {key}: tuning item-item KNN...")
        knn_best = None
        for neigh_k in knn_candidates:
            # Build similarity using train only (tuning)
            neighbors, sims = train_item_item_knn(
                train_df, num_users, num_items, neighbor_k=neigh_k
            )
            knn_val_metrics = evaluate_knn_ranking_for_test_negs(
                test_negs_df=val_negs_df.rename(columns={"pos_movie_index": "pos_movie_index"}),  # keep same schema
                test_df=val_df,
                neighbors=neighbors,
                sims=sims,
                train_df_for_user_history=train_df,
                ks=ks,
            )
            ndcg10 = float(knn_val_metrics.get("ndcg@10", 0.0))
            print(f"[knn] {key}: neigh_k={neigh_k} val_ndcg@10={ndcg10:.6f}")
            if knn_best is None or ndcg10 > knn_best["ndcg10"]:
                knn_best = {"neigh_k": neigh_k, "ndcg10": ndcg10, "val_metrics": knn_val_metrics}

        # Retrain KNN on train+val and evaluate on test
        print(f"[knn] {key}: training final KNN with neigh_k={knn_best['neigh_k']}...")
        trainval_df = pd.concat([train_df, val_df], ignore_index=True)
        neighbors_final, sims_final = train_item_item_knn(
            trainval_df, num_users, num_items, neighbor_k=knn_best["neigh_k"]
        )
        knn_test_metrics = evaluate_knn_ranking_for_test_negs(
            test_negs_df=test_negs_df,
            test_df=test_df,
            neighbors=neighbors_final,
            sims=sims_final,
            train_df_for_user_history=trainval_df,
            ks=ks,
        )

        # KNN rating prediction RMSE/MAE on test.
        # (Use the same scoring function with candidate = [i] per test row.)
        user_hist = {}
        for u0, i0, r0 in zip(
            trainval_df["user_index"].to_numpy(dtype=np.int32, copy=False),
            trainval_df["movie_index"].to_numpy(dtype=np.int32, copy=False),
            trainval_df["rating"].to_numpy(dtype=np.float32, copy=False),
        ):
            user_hist.setdefault(int(u0), {})[int(i0)] = float(r0)

        # Precompute per-user mean for fallback.
        user_mean = {}
        for u0, d in user_hist.items():
            user_mean[u0] = float(np.mean(list(d.values()))) if d else float(trainval_df["rating"].mean())
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

        knn_results = {
            "neighbor_k": knn_best["neigh_k"],
            "val_ranking": knn_best["val_metrics"],
            "test_ranking": knn_test_metrics,
            "test_rating": {"rmse": rmse_knn, "mae": mae_knn},
        }

        # ===== ALS tuning on val ranking =====
        print(f"[als] {key}: tuning ALS rank/reg on val...")
        als_best = None
        for rank in als_ranks:
            for reg in als_regs:
                out = train_and_evaluate_spark_als_residual_biases(
                    spark=spark,
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    val_negs_df=val_negs_df[["user_index", "pos_movie_index", "neg_movie_index"]],
                    test_negs_df=test_negs_df[["user_index", "pos_movie_index", "neg_movie_index"]],
                    num_users=num_users,
                    num_items=num_items,
                    rank=rank,
                    reg_param=reg,
                    max_iter=args.als_max_iter,
                    bias_lambda=args.als_bias_lambda,
                    seed=args.seed,
                    ks=ks,
                )
                ndcg10 = float(out.get("val_ndcg@10", 0.0))
                print(f"[als] {key}: rank={rank} reg={reg} val_ndcg@10={ndcg10:.6f}")
                if als_best is None or ndcg10 > als_best["val_ndcg10"]:
                    als_best = {"rank": rank, "reg": reg, "val_ndcg10": ndcg10, "out": out}

        # Retrain best ALS on train+val and evaluate on test
        print(f"[als] {key}: training final ALS rank={als_best['rank']} reg={als_best['reg']}...")
        # We can reuse training function by providing val_df=test_df? But we want tune on val then final on train+val.
        # We'll do: final train_df = concat(train,val), use val_df as original val only for metrics function; it won't matter for test metrics.
        trainval_df = pd.concat([train_df, val_df], ignore_index=True)
        # For final evaluation, we keep val as original val, but bias/ALS trained on train+val.
        final_out = train_and_evaluate_spark_als_residual_biases(
            spark=spark,
            train_df=trainval_df,
            val_df=val_df,
            test_df=test_df,
            val_negs_df=val_negs_df[["user_index", "pos_movie_index", "neg_movie_index"]],
            test_negs_df=test_negs_df[["user_index", "pos_movie_index", "neg_movie_index"]],
            num_users=num_users,
            num_items=num_items,
            rank=als_best["rank"],
            reg_param=als_best["reg"],
            max_iter=args.als_max_iter,
            bias_lambda=args.als_bias_lambda,
            seed=args.seed,
            ks=ks,
        )

        als_results = {
            "best_rank": als_best["rank"],
            "best_reg_param": als_best["reg"],
            "val_ranking": final_out["val_ranking"],
            "test_ranking": final_out["test_ranking"],
            "test_rating": final_out["test_rating"],
        }

        out_dir = RESULTS_DIR / key
        out_dir.mkdir(parents=True, exist_ok=True)
        results = {"dataset": key, "knn": knn_results, "als": als_results, "ks": ks}
        with open(out_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"[done] {key}: wrote results to {out_dir / 'results.json'}")

    spark.stop()


if __name__ == "__main__":
    main()

