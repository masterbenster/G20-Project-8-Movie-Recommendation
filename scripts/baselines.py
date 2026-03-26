import argparse
import json
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = DATA_DIR / "results" / "baselines"


@dataclass(frozen=True)
class DatasetSpec:
    key: str  # "1m" or "10m"
    extract_dir: pathlib.Path


def _read_csv_gz(path: pathlib.Path, usecols=None, dtypes=None) -> pd.DataFrame:
    if dtypes is None:
        return pd.read_csv(path, compression="gzip", usecols=usecols)
    return pd.read_csv(path, compression="gzip", usecols=usecols, dtype=dtypes)


def _rmse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(np.square(err))))
    mae = float(np.mean(np.abs(err)))
    return rmse, mae


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


def evaluate_topn_onepos_per_user(
    test_negs_df: pd.DataFrame,
    score_fn,
    ks: list[int],
) -> dict[str, float]:
    """
    Assumes test_negs_df contains exactly one positive item per user, plus negatives.
    Metrics are averaged across users.
    """
    u_arr = test_negs_df["user_index"].to_numpy(dtype=np.int32, copy=False)
    pos_arr = test_negs_df["pos_movie_index"].to_numpy(dtype=np.int32, copy=False)
    neg_arr = test_negs_df["neg_movie_index"].to_numpy(dtype=np.int32, copy=False)

    # Ensure grouped by user_index for efficient iteration.
    if not np.all(u_arr[1:] >= u_arr[:-1]):
        idx = np.argsort(u_arr, kind="stable")
        u_arr = u_arr[idx]
        pos_arr = pos_arr[idx]
        neg_arr = neg_arr[idx]

    # Group boundaries.
    boundaries = np.flatnonzero(u_arr[1:] != u_arr[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(u_arr)]))
    num_users = len(starts)

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
        **{f"precision@{k}": prec_sum[k] / num_users for k in ks},
        **{f"recall@{k}": rec_sum[k] / num_users for k in ks},
        **{f"ndcg@{k}": ndcg_sum[k] / num_users for k in ks},
    }


def load_genre_tag_fallback_scores(dataset_spec: DatasetSpec, movie_map_df: pd.DataFrame, train_df: pd.DataFrame):
    """
    Builds a GLOBAL genre/tag based score per movie_index for cold-start fallback.
    (User-independent baseline.)
    """
    movie_map = movie_map_df[["movieId_raw", "movie_index"]].copy()
    movie_map["movie_index"] = movie_map["movie_index"].astype(np.int32)

    # Popularity part: counts of interactions in training.
    counts = np.bincount(train_df["movie_index"].to_numpy(dtype=np.int32, copy=False), minlength=int(movie_map["movie_index"].max()) + 1)
    pop_score = np.log1p(counts).astype(np.float32)

    # Genre part: based on MovieLens genre labels in movies.dat.
    movies_dat = dataset_spec.extract_dir / "movies.dat"
    if not movies_dat.exists():
        return pop_score

    movies_df = pd.read_csv(
        movies_dat,
        sep="::",
        engine="python",
        header=None,
        names=["movieId_raw", "title", "genres"],
        dtype={"movieId_raw": np.int32, "title": str, "genres": str},
        encoding="latin-1",
    )
    movies_df = movies_df.merge(movie_map, on="movieId_raw", how="inner")

    # Compute global genre weights.
    genre_counts = {}
    movie_genre_weights = np.zeros(len(movie_map), dtype=np.float32)
    # Small genre set; Python loops over movies is fine.
    # (movies.dat has ~3-5k rows for 1M and ~10k for 10M; manageable.)
    for _, row in movies_df.iterrows():
        genres = str(row["genres"]).split("|") if row["genres"] else []
        for g in genres:
            genre_counts[g] = genre_counts.get(g, 0) + 1

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
        tags_df = pd.read_csv(
            tags_dat,
            sep="::",
            engine="python",
            header=None,
            names=["userId_raw", "movieId_raw", "tag", "timestamp"],
            dtype={"movieId_raw": np.int32, "userId_raw": np.int32, "tag": str, "timestamp": np.int64},
            encoding="latin-1",
        )
        tag_counts = tags_df.groupby("movieId_raw").size().reset_index(name="tag_count")
        tag_counts = tag_counts.merge(movie_map, on="movieId_raw", how="inner")
        tag_score = np.zeros(len(movie_map), dtype=np.float32)
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
    composite = (alpha * pop_scaled + beta * genre_scaled + gamma * tag_scaled).astype(np.float32)
    return composite


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: Baseline models + evaluation.")
    parser.add_argument("--dataset", choices=["1m", "10m", "both"], default="both")
    parser.add_argument("--num-negatives", type=int, default=99, help="Expected negative count per user.")
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
        train_df = _read_csv_gz(
            processed_dir / "train.csv.gz",
            dtypes={"user_index": np.int32, "movie_index": np.int32, "rating": np.float32, "timestamp": np.int64},
        )
        test_df = _read_csv_gz(
            processed_dir / "test.csv.gz",
            dtypes={"user_index": np.int32, "movie_index": np.int32, "rating": np.float32, "timestamp": np.int64},
        )
        movie_map_df = pd.read_csv(processed_dir / "movie_map.csv")

        test_negs_df = _read_csv_gz(
            processed_dir / "test_negs.csv.gz",
            dtypes={"user_index": np.int32, "pos_movie_index": np.int32, "neg_movie_index": np.int32},
        )
        if int(test_negs_df["neg_movie_index"].nunique()) < 10:
            print(f"[warn] {key}: test_negs looks suspiciously small.")

        print(f"[bias] {key}: training user/movie bias model...")
        mu_train, b_u, b_i = train_user_movie_bias(
            train_df=train_df,
            num_users=num_users,
            num_movies=num_movies,
            lambda_reg=args.lambda_reg,
            num_iters=args.num_iters,
        )

        # Rating prediction evaluation.
        y_true = test_df["rating"].to_numpy(dtype=np.float32, copy=False)
        u_test = test_df["user_index"].to_numpy(dtype=np.int32, copy=False)
        i_test = test_df["movie_index"].to_numpy(dtype=np.int32, copy=False)

        pred_global = np.full_like(y_true, fill_value=mu_train, dtype=np.float32)
        pred_bias = (mu_train + b_u[u_test] + b_i[i_test]).astype(np.float32, copy=False)

        rmse_global, mae_global = _rmse_mae(y_true, pred_global)
        rmse_bias, mae_bias = _rmse_mae(y_true, pred_bias)

        rating_metrics = {
            "global_mean": {"rmse": rmse_global, "mae": mae_global},
            "user_movie_bias": {"rmse": rmse_bias, "mae": mae_bias},
        }

        # Popularity baseline scores.
        pop_counts = np.bincount(train_df["movie_index"].to_numpy(dtype=np.int32, copy=False), minlength=num_movies).astype(np.float32)
        pop_scores = np.log1p(pop_counts).astype(np.float32)

        # Genre/tag fallback (user-independent global score).
        print(f"[fallback] {key}: building genre/tag fallback scores...")
        genre_tag_scores = load_genre_tag_fallback_scores(spec, movie_map_df, train_df)

        print(f"[rank] {key}: evaluating top-N metrics...")
        # Each score_fn returns scores for candidate movie indices; shape = (num_candidates,).
        def popularity_score_fn(u: int, candidates: np.ndarray) -> np.ndarray:
            return pop_scores[candidates]

        def bias_score_fn(u: int, candidates: np.ndarray) -> np.ndarray:
            # bias score only depends on user u and movie candidates.
            return (mu_train + b_u[u] + b_i[candidates]).astype(np.float32, copy=False)

        def genre_tag_score_fn(u: int, candidates: np.ndarray) -> np.ndarray:
            return genre_tag_scores[candidates]

        topn_metrics = {}
        topn_metrics["popularity"] = evaluate_topn_onepos_per_user(test_negs_df, popularity_score_fn, ks)
        topn_metrics["user_movie_bias_ranking"] = evaluate_topn_onepos_per_user(test_negs_df, bias_score_fn, ks)
        topn_metrics["genre_tag_fallback"] = evaluate_topn_onepos_per_user(test_negs_df, genre_tag_score_fn, ks)

        results = {
            "dataset": key,
            "ks": ks,
            "rating_metrics": rating_metrics,
            "topn_metrics": topn_metrics,
            "model_artifacts": {
                "bias_model": {
                    "mu_train": mu_train,
                    "lambda_reg": args.lambda_reg,
                    "num_iters": args.num_iters,
                    "b_u_path": str(out_dir / "bias_bu.npy"),
                    "b_i_path": str(out_dir / "bias_bi.npy"),
                },
                "popularity_scores_path": str(out_dir / "popularity_scores.npy"),
                "genre_tag_scores_path": str(out_dir / "genre_tag_scores.npy"),
            },
        }
        with open(out_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        np.save(out_dir / "bias_bu.npy", b_u)
        np.save(out_dir / "bias_bi.npy", b_i)
        np.save(out_dir / "popularity_scores.npy", pop_scores)
        np.save(out_dir / "genre_tag_scores.npy", genre_tag_scores)

        print(f"[done] {key}: wrote baselines to {out_dir}")


if __name__ == "__main__":
    main()

