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
RESULTS_DIR = DATA_DIR / "results" / "baselines"


@dataclass(frozen=True)
class DatasetSpec:
    key: str  # "1m" or "10m"
    extract_dir: pathlib.Path


def _rmse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(np.square(err))))
    mae = float(np.mean(np.abs(err)))
    return rmse, mae


def build_final_train_df(train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    """
    Final baseline evaluation uses train+val after validation-time model selection.
    """
    return pd.concat([train_df, val_df], ignore_index=True)


def train_user_movie_bias(
    train_df: pd.DataFrame,
    num_users: int,
    num_movies: int,
    lambda_reg: float = 25.0,
    num_iters: int = 10,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Fits: r_ui ~= mu + b_u[u] + b_i[i], where b_u and b_i are regularized.
    Uses iterative coordinate updates with bincount for speed.
    """
    # Ensure we have compact dtypes for numpy math.
    U = train_df["user_index"].to_numpy(dtype=np.int32, copy=False)
    I = train_df["movie_index"].to_numpy(dtype=np.int32, copy=False)
    R = train_df["rating"].to_numpy(dtype=np.float32, copy=False)

    mu = float(np.mean(R, dtype=np.float64))
    b_u = np.zeros(num_users, dtype=np.float32)
    b_i = np.zeros(num_movies, dtype=np.float32)

    count_u = np.bincount(U, minlength=num_users).astype(np.float32)
    count_i = np.bincount(I, minlength=num_movies).astype(np.float32)

    # Iterative updates.
    for _ in range(num_iters):
        # Update user biases.
        # residual = r_ui - mu - b_i[item]
        res_u = R - mu - b_i[I]
        sum_res_u = np.bincount(U, weights=res_u, minlength=num_users).astype(np.float32)
        b_u = sum_res_u / (lambda_reg + count_u)

        # Update item biases.
        res_i = R - mu - b_u[U]
        sum_res_i = np.bincount(I, weights=res_i, minlength=num_movies).astype(np.float32)
        b_i = sum_res_i / (lambda_reg + count_i)

    return mu, b_u, b_i


def train_user_mean_predictor(train_df: pd.DataFrame, num_users: int, fallback: float) -> np.ndarray:
    u = train_df["user_index"].to_numpy(dtype=np.int32, copy=False)
    r = train_df["rating"].to_numpy(dtype=np.float32, copy=False)
    count_u = np.bincount(u, minlength=num_users).astype(np.float32)
    sum_u = np.bincount(u, weights=r, minlength=num_users).astype(np.float32)
    out = np.full(num_users, fill_value=float(fallback), dtype=np.float32)
    seen = count_u > 0
    out[seen] = sum_u[seen] / count_u[seen]
    return out


def train_item_mean_predictor(train_df: pd.DataFrame, num_movies: int, fallback: float) -> np.ndarray:
    i = train_df["movie_index"].to_numpy(dtype=np.int32, copy=False)
    r = train_df["rating"].to_numpy(dtype=np.float32, copy=False)
    count_i = np.bincount(i, minlength=num_movies).astype(np.float32)
    sum_i = np.bincount(i, weights=r, minlength=num_movies).astype(np.float32)
    out = np.full(num_movies, fill_value=float(fallback), dtype=np.float32)
    seen = count_i > 0
    out[seen] = sum_i[seen] / count_i[seen]
    return out


def _sorted_rank_and_scores(candidates: np.ndarray, scores: np.ndarray) -> int:
    """
    Returns 1-indexed rank of candidates[0] (the "positive") after sorting:
    primary: higher score first
    secondary: higher movie_index first (deterministic tie-break).
    """
    cand = candidates.astype(np.int32, copy=False)
    scr = scores.astype(np.float32, copy=False)
    # lexsort sorts by last key first; we want primary = -scr, secondary = -cand.
    order = np.lexsort((-cand, -scr))
    pos_in_order = int(np.where(order == 0)[0][0])
    return pos_in_order + 1


def evaluate_topn_candidate_groups(
    candidate_df: pd.DataFrame,
    score_fn,
    ks: list[int],
) -> dict[str, float]:
    """
    Assumes candidate_df contains one positive movie per (user_index, pos_movie_index)
    candidate group and multiple sampled negatives for that same event.
    Metrics are averaged across candidate groups.
    """
    event_arr = candidate_df["event_id"].to_numpy(dtype=np.int64, copy=False)
    u_arr = candidate_df["user_index"].to_numpy(dtype=np.int32, copy=False)
    pos_arr = candidate_df["pos_movie_index"].to_numpy(dtype=np.int32, copy=False)
    neg_arr = candidate_df["neg_movie_index"].to_numpy(dtype=np.int32, copy=False)

    order = np.lexsort((neg_arr, pos_arr, u_arr, event_arr))
    event_arr = event_arr[order]
    u_arr = u_arr[order]
    pos_arr = pos_arr[order]
    neg_arr = neg_arr[order]

    # Group boundaries over event_id.
    boundaries = np.flatnonzero(event_arr[1:] != event_arr[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(u_arr)]))
    num_events = len(starts)

    max_k = max(ks)
    # Accumulators.
    prec_sum = {k: 0.0 for k in ks}
    rec_sum = {k: 0.0 for k in ks}
    ndcg_sum = {k: 0.0 for k in ks}

    for s, e in zip(starts, ends):
        u = int(u_arr[s])
        pos = int(pos_arr[s])
        negs = neg_arr[s:e]

        if len(negs) + 1 < max_k:
            # Should never happen with consistent negative sampling, but guard anyway.
            continue

        candidates = np.concatenate((np.array([pos], dtype=np.int32), negs))
        scores = score_fn(u, candidates)

        rank_pos = _sorted_rank_and_scores(candidates, scores)

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


def _read_movielens_dat(path: pathlib.Path, *, names: list[str], dtypes: dict[str, object]) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(
                path,
                sep="::",
                engine="python",
                header=None,
                names=names,
                dtype=dtypes,
                encoding=encoding,
            )
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error if last_error is not None else RuntimeError(f"Unable to read MovieLens data file: {path}")


def build_metadata_rankers(
    dataset_spec: DatasetSpec,
    movie_map_df: pd.DataFrame,
    train_df: pd.DataFrame,
    *,
    num_users: int,
):
    """
    Builds:
    - a global metadata fallback score per movie_index
    - a user-conditioned genre profile scorer backed by movie metadata
    """
    movie_map = movie_map_df[["movieId_raw", "movie_index"]].copy()
    movie_map["movie_index"] = movie_map["movie_index"].astype(np.int32)
    num_movies = int(movie_map["movie_index"].max()) + 1

    # Popularity part: counts of interactions in training.
    counts = np.bincount(
        train_df["movie_index"].to_numpy(dtype=np.int32, copy=False),
        minlength=num_movies,
    )
    pop_score = np.log1p(counts).astype(np.float32)

    # Genre part: based on MovieLens genre labels in movies.dat.
    movies_dat = dataset_spec.extract_dir / "movies.dat"
    if not movies_dat.exists():
        def _global_only(u: int, candidates: np.ndarray) -> np.ndarray:
            return pop_score[candidates]

        return pop_score, _global_only

    movies_df = _read_movielens_dat(
        movies_dat,
        names=["movieId_raw", "title", "genres"],
        dtypes={"movieId_raw": np.int32, "title": str, "genres": str},
    )
    movies_df = movies_df.merge(movie_map, on="movieId_raw", how="inner")

    # Compute global genre weights.
    genre_counts = {}
    movie_genre_weights = np.zeros(num_movies, dtype=np.float32)
    genre_pairs: list[tuple[int, str]] = []
    # Small genre set; Python loops over movies is fine.
    # (movies.dat has ~3-5k rows for 1M and ~10k for 10M; manageable.)
    for _, row in movies_df.iterrows():
        genres = str(row["genres"]).split("|") if row["genres"] else []
        for g in genres:
            genre_counts[g] = genre_counts.get(g, 0) + 1
            genre_pairs.append((int(row["movie_index"]), g))

    # Weight function: log(1 + count) so common genres get higher score.
    genre_weight = {g: np.log1p(c) for g, c in genre_counts.items()}
    for _, row in movies_df.iterrows():
        genres = str(row["genres"]).split("|") if row["genres"] else []
        w = float(sum(genre_weight.get(g, 0.0) for g in genres))
        movie_genre_weights[int(row["movie_index"])] = w

    genre_score = movie_genre_weights

    # Tag part: based on tags.dat where available (mainly 10M).
    tags_dat = dataset_spec.extract_dir / "tags.dat"
    if not tags_dat.exists():
        tag_score = np.zeros_like(genre_score, dtype=np.float32)
    else:
        tags_df = _read_movielens_dat(
            tags_dat,
            names=["userId_raw", "movieId_raw", "tag", "timestamp"],
            dtypes={"movieId_raw": np.int32, "userId_raw": np.int32, "tag": str, "timestamp": np.int64},
        )
        tag_counts = tags_df.groupby("movieId_raw").size().reset_index(name="tag_count")
        tag_counts = tag_counts.merge(movie_map, on="movieId_raw", how="inner")
        tag_score = np.zeros(num_movies, dtype=np.float32)
        # Fill by movie_index.
        for _, row in tag_counts.iterrows():
            tag_score[int(row["movie_index"])] = float(row["tag_count"])
        tag_score = np.log1p(tag_score).astype(np.float32)

    # Normalize genre score to a similar scale as pop_score and tag_score.
    # Avoid external deps; simple min-max-like scaling.
    def _scale(x: np.ndarray) -> np.ndarray:
        denom = float(x.max() - x.min())
        if denom <= 1e-12:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - float(x.min())) / denom).astype(np.float32)

    genre_scaled = _scale(genre_score)
    tag_scaled = _scale(tag_score)
    pop_scaled = _scale(pop_score)

    # Composite score weights for fallback.
    alpha, beta, gamma = 0.6, 0.2, 0.2
    global_composite = (alpha * pop_scaled + beta * genre_scaled + gamma * tag_scaled).astype(np.float32)

    genre_vocab = sorted(genre_counts)
    genre_index = {g: idx for idx, g in enumerate(genre_vocab)}
    genre_rows = []
    genre_cols = []
    for movie_index, genre_name in genre_pairs:
        genre_rows.append(movie_index)
        genre_cols.append(genre_index[genre_name])
    genre_matrix = sparse.csr_matrix(
        (np.ones(len(genre_rows), dtype=np.float32), (genre_rows, genre_cols)),
        shape=(num_movies, len(genre_vocab)),
        dtype=np.float32,
    )

    train_u = train_df["user_index"].to_numpy(dtype=np.int32, copy=False)
    train_i = train_df["movie_index"].to_numpy(dtype=np.int32, copy=False)
    train_r = train_df["rating"].to_numpy(dtype=np.float32, copy=False)
    user_pref_weight = np.maximum(train_r - 3.0, 0.0).astype(np.float32)
    user_item_pref = sparse.csr_matrix(
        (user_pref_weight, (train_u, train_i)),
        shape=(num_users, num_movies),
        dtype=np.float32,
    )
    user_genre_profiles = user_item_pref @ genre_matrix
    user_profile_norms = np.sqrt(user_genre_profiles.multiply(user_genre_profiles).sum(axis=1)).A1.astype(np.float32)

    def _genre_tag_profile(u: int, candidates: np.ndarray) -> np.ndarray:
        base = global_composite[candidates].astype(np.float32, copy=False)
        if u < 0 or u >= num_users:
            return base

        profile = user_genre_profiles.getrow(u)
        profile_norm = float(user_profile_norms[u])
        if profile.nnz == 0 or profile_norm <= 1e-12:
            return base

        genre_pref = (genre_matrix[candidates] @ profile.T).toarray().reshape(-1).astype(np.float32, copy=False)
        genre_pref = genre_pref / profile_norm
        return (0.65 * genre_pref + 0.25 * pop_scaled[candidates] + 0.10 * tag_scaled[candidates]).astype(
            np.float32,
            copy=False,
        )

    return global_composite, _genre_tag_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: Baseline models + evaluation.")
    parser.add_argument("--dataset", choices=["1m", "10m", "both"], default="both")
    parser.add_argument("--num-negatives", type=int, default=99, help="Negative candidates per held-out event.")
    parser.add_argument(
        "--max-ranking-events",
        type=int,
        default=None,
        help="Maximum held-out events to evaluate for ranking. Default: 20000 for both 1m and 10m.",
    )
    parser.add_argument("--ks", type=str, default="5,10,20")
    parser.add_argument("--lambda-reg", type=float, default=25.0)
    parser.add_argument("--num-iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)  # kept for reproducibility across potential future changes
    args = parser.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x]

    specs = {
        "1m": DatasetSpec(key="1m", extract_dir=RAW_DIR / "ml-1m"),
        "10m": DatasetSpec(key="10m", extract_dir=RAW_DIR / "ml-10M100K"),
    }
    targets = ["1m", "10m"] if args.dataset == "both" else [args.dataset]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for key in targets:
        spec = specs[key]
        processed_dir = PROCESSED_DIR / key
        meta = json.load(open(processed_dir / "meta.json", "r"))
        num_users = int(meta["num_users"])
        num_movies = int(meta["num_movies"])

        out_dir = RESULTS_DIR / key
        out_dir.mkdir(parents=True, exist_ok=True)

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
        movie_map_df = pd.read_csv(processed_dir / "movie_map.csv")
        trainval_df = build_final_train_df(train_df, val_df)
        max_ranking_events = default_max_ranking_events(key) if args.max_ranking_events is None else args.max_ranking_events
        test_events_df, test_negs_df, ranking_eval_summary = build_candidate_groups(
            split_df=test_df,
            rated_source_df=ratings_clean_df,
            num_movies=num_movies,
            num_negatives=args.num_negatives,
            seed=args.seed + 200,
            max_events=max_ranking_events,
        )

        print(f"[bias] {key}: training final baseline models on train+val...")
        mu_train, b_u, b_i = train_user_movie_bias(
            train_df=trainval_df,
            num_users=num_users,
            num_movies=num_movies,
            lambda_reg=args.lambda_reg,
            num_iters=args.num_iters,
        )
        user_mean = train_user_mean_predictor(train_df=trainval_df, num_users=num_users, fallback=mu_train)
        item_mean = train_item_mean_predictor(train_df=trainval_df, num_movies=num_movies, fallback=mu_train)

        # Rating prediction evaluation.
        y_true = test_df["rating"].to_numpy(dtype=np.float32, copy=False)
        u_test = test_df["user_index"].to_numpy(dtype=np.int32, copy=False)
        i_test = test_df["movie_index"].to_numpy(dtype=np.int32, copy=False)

        pred_global = np.full_like(y_true, fill_value=mu_train, dtype=np.float32)
        pred_user_mean = user_mean[u_test].astype(np.float32, copy=False)
        pred_item_mean = item_mean[i_test].astype(np.float32, copy=False)
        pred_bias = (mu_train + b_u[u_test] + b_i[i_test]).astype(np.float32, copy=False)

        rmse_global, mae_global = _rmse_mae(y_true, pred_global)
        rmse_user_mean, mae_user_mean = _rmse_mae(y_true, pred_user_mean)
        rmse_item_mean, mae_item_mean = _rmse_mae(y_true, pred_item_mean)
        rmse_bias, mae_bias = _rmse_mae(y_true, pred_bias)

        rating_metrics = {
            "global_mean": {"rmse": rmse_global, "mae": mae_global},
            "user_mean": {"rmse": rmse_user_mean, "mae": mae_user_mean},
            "item_mean": {"rmse": rmse_item_mean, "mae": mae_item_mean},
            "user_movie_bias": {"rmse": rmse_bias, "mae": mae_bias},
        }

        # Popularity baseline scores.
        pop_counts = np.bincount(trainval_df["movie_index"].to_numpy(dtype=np.int32, copy=False), minlength=num_movies).astype(np.float32)
        pop_scores = np.log1p(pop_counts).astype(np.float32)

        # Genre/tag fallback (user-independent global score).
        print(f"[fallback] {key}: building genre/tag fallback scores...")
        global_metadata_scores, genre_tag_profile_score_fn = build_metadata_rankers(
            spec,
            movie_map_df,
            trainval_df,
            num_users=num_users,
        )

        print(f"[rank] {key}: evaluating top-N metrics...")
        # Each score_fn returns scores for candidate movie indices; shape = (num_candidates,).
        def popularity_score_fn(u: int, candidates: np.ndarray) -> np.ndarray:
            return pop_scores[candidates]

        def bias_score_fn(u: int, candidates: np.ndarray) -> np.ndarray:
            # bias score only depends on user u and movie candidates.
            return (mu_train + b_u[u] + b_i[candidates]).astype(np.float32, copy=False)

        def metadata_global_score_fn(u: int, candidates: np.ndarray) -> np.ndarray:
            return global_metadata_scores[candidates]

        topn_metrics = {}
        topn_metrics["popularity"] = evaluate_topn_candidate_groups(test_negs_df, popularity_score_fn, ks)
        topn_metrics["user_movie_bias_ranking"] = evaluate_topn_candidate_groups(test_negs_df, bias_score_fn, ks)
        topn_metrics["metadata_global"] = evaluate_topn_candidate_groups(test_negs_df, metadata_global_score_fn, ks)
        topn_metrics["genre_tag_profile"] = evaluate_topn_candidate_groups(test_negs_df, genre_tag_profile_score_fn, ks)

        results = {
            "dataset": key,
            "ks": ks,
            "training_data": "trainval",
            "rating_metrics": rating_metrics,
            "topn_metrics": topn_metrics,
            "model_artifacts": {
                "bias_model": {
                    "mu_train": mu_train,
                    "lambda_reg": args.lambda_reg,
                    "num_iters": args.num_iters,
                    "training_data": "trainval",
                    "b_u_path": str(out_dir / "bias_bu.npy"),
                    "b_i_path": str(out_dir / "bias_bi.npy"),
                },
                "popularity_scores_path": str(out_dir / "popularity_scores.npy"),
                "metadata_global_scores_path": str(out_dir / "metadata_global_scores.npy"),
            },
            "ranking_eval": ranking_eval_summary,
        }
        with open(out_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        np.save(out_dir / "bias_bu.npy", b_u)
        np.save(out_dir / "bias_bi.npy", b_i)
        np.save(out_dir / "popularity_scores.npy", pop_scores)
        np.save(out_dir / "metadata_global_scores.npy", global_metadata_scores)

        print(f"[done] {key}: wrote baselines to {out_dir}")


if __name__ == "__main__":
    main()
