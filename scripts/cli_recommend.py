import argparse
import json
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = DATA_DIR / "models" / "als"


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

def _parse_movie_ratings(pairs: str) -> list[tuple[int, float]]:
    """
    Parse comma-separated "movieId:rating" pairs.

    Example: "1:5,260:3.5,1193:4"
    """
    out: list[tuple[int, float]] = []
    for part in pairs.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid rating pair (expected movieId:rating): {part!r}")
        movie_id_raw_s, rating_s = part.split(":", 1)
        movie_id_raw = int(movie_id_raw_s.strip())
        rating = float(rating_s.strip())
        out.append((movie_id_raw, rating))
    return out


def _read_ratings_file(path: pathlib.Path) -> str:
    """
    Read cold-start ratings from a text file.

    Supported formats:
    - comma-separated pairs: "1:5,260:3.5,1193:4"
    - newline-separated pairs:
        1:5
        260:3.5
        1193:4
    """
    content = path.read_text(encoding="utf-8")
    # Normalize whitespace and allow newlines to act like commas.
    content = content.replace("\n", ",").replace("\r", ",")
    return content.strip()


def _estimate_new_user_from_ratings(
    *,
    movie_indices: np.ndarray,  # (n,)
    ratings: np.ndarray,  # (n,)
    mu: float,
    bias_bi: np.ndarray,  # (num_items,)
    item_factors: np.ndarray,  # (num_items, rank)
    cold_start_reg: float,
) -> tuple[float, np.ndarray]:
    """
    Estimate new-user parameters for the ALS residual model:

      rating_pred = mu + b_u + b_i + u^T v_i

    Unknowns are (b_u, u). We solve a ridge-regularized least squares problem:
      y = b_u + u^T v_i
    where y = rating - mu - b_i.
    """
    if movie_indices.size == 0:
        raise ValueError("No valid movies in --new-user-ratings (after mapping).")
    if movie_indices.shape[0] != ratings.shape[0]:
        raise ValueError("Internal error: movie_indices and ratings length mismatch.")

    rank = int(item_factors.shape[1])
    n = int(movie_indices.shape[0])

    # Targets with biases removed.
    y = ratings.astype(np.float64, copy=False) - float(mu) - bias_bi[movie_indices].astype(np.float64, copy=False)

    # Design matrix X = [1, V_selected] so that y ~= X @ [b_u, u...].
    V = item_factors[movie_indices].astype(np.float64, copy=False)  # (n, rank)
    X = np.empty((n, rank + 1), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1:] = V

    # Ridge penalty should not regularize the bias term b_u.
    reg = float(cold_start_reg)
    if reg < 0:
        raise ValueError(f"--cold-start-reg must be >= 0, got {reg}")

    # (X^T X + diag([0, reg, reg, ...])) w = X^T y
    A = X.T @ X
    reg_vec = np.concatenate(([0.0], np.full(rank, reg, dtype=np.float64)))
    A_reg = A + np.diag(reg_vec)
    b = X.T @ y

    w = np.linalg.solve(A_reg, b).astype(np.float64, copy=False)  # (rank+1,)
    b_u_new = float(w[0])
    u_new = w[1:].astype(np.float32, copy=False)
    return b_u_new, u_new


def _dataset_raw_dir(dataset_key: str) -> pathlib.Path:
    # Matches how `prepare_data.py` unzips MovieLens zips.
    if dataset_key == "1m":
        return RAW_DIR / "ml-1m"
    if dataset_key == "10m":
        return RAW_DIR / "ml-10M100K"
    raise ValueError(f"Unknown dataset: {dataset_key}")


def _load_user_index_map(processed_dir: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(processed_dir / "user_map.csv")


def _load_movie_index_map(processed_dir: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(processed_dir / "movie_map.csv")


def _load_movie_titles(dataset_key: str, processed_dir: pathlib.Path) -> tuple[list[str], list[str]]:
    """
    Returns:
      titles[movie_index], genres[movie_index]
    """
    raw_dir = _dataset_raw_dir(dataset_key)
    movies_path = raw_dir / "movies.dat"
    if not movies_path.exists():
        raise FileNotFoundError(f"Missing MovieLens movies metadata: {movies_path}")

    movies_df = _read_movielens_dat(
        movies_path,
        names=["movieId_raw", "title", "genres"],
        dtypes={"movieId_raw": np.int32, "title": str, "genres": str},
    )
    movie_map = _load_movie_index_map(processed_dir)
    movies_df = movies_df.merge(movie_map, on="movieId_raw", how="inner")

    # movie_index is contiguous [0..num_movies). Build arrays by index.
    num_movies = int(movie_map["movie_index"].max()) + 1
    titles = [""] * num_movies
    genres = [""] * num_movies
    for _, row in movies_df.iterrows():
        mi = int(row["movie_index"])
        titles[mi] = str(row["title"])
        genres[mi] = str(row["genres"])
    return titles, genres


def _load_rated_movies_for_user(history_csv_gz: pathlib.Path, user_index: int, chunksize: int = 200000) -> set[int]:
    rated: set[int] = set()
    # Stream through a processed history file and collect movie_index rows for this user_index.
    # This avoids loading the full dataset into RAM.
    dtypes = {"user_index": np.int32, "movie_index": np.int32}
    for chunk in pd.read_csv(
        history_csv_gz,
        compression="gzip",
        usecols=["user_index", "movie_index"],
        dtype=dtypes,
        chunksize=chunksize,
    ):
        sub = chunk[chunk["user_index"] == user_index]
        if len(sub) > 0:
            rated.update(sub["movie_index"].astype(int).tolist())
    return rated


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-N movie recommendations CLI (ALS residual factors).")
    parser.add_argument("--dataset", choices=["1m", "10m"], required=True)
    parser.add_argument("--user-id", "--user", dest="user_id", type=int, default=None, help="MovieLens raw userId.")
    parser.add_argument(
        "--user-index",
        type=int,
        default=None,
        help="Contiguous user_index used by processed data (mostly for debugging).",
    )
    parser.add_argument(
        "--new-user-ratings",
        "--ratings",
        dest="new_user_ratings",
        type=str,
        default=None,
        help=(
            "Cold-start mode: comma-separated 'movieId:rating' pairs for a new user, "
            "e.g. '1:5,260:3.5,1193:4'. Mutually exclusive with --user-id/--user-index."
        ),
    )
    parser.add_argument(
        "--ratings-file",
        type=str,
        default=None,
        help="Path to a text file containing 'movieId:rating' pairs (comma- or newline-separated).",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format. 'text' is human-readable; 'json' is machine-readable.",
    )
    parser.add_argument("--max-title-chars", type=int, default=120, help="Truncate long titles for text output.")
    parser.add_argument("--score-decimals", type=int, default=4, help="Number of decimal places for the score.")
    parser.add_argument("--show-movie-index", action="store_true", default=False, help="Include internal movie_index in JSON/text.")
    parser.add_argument(
        "--exclude-rated",
        dest="exclude_rated",
        action="store_true",
        default=True,
        help="Exclude seen items for the user. Enabled by default.",
    )
    parser.add_argument(
        "--include-seen",
        dest="exclude_rated",
        action="store_false",
        help="Allow recommendations that include seen items.",
    )
    parser.add_argument(
        "--exclude-seen-scope",
        choices=["train", "all"],
        default="all",
        help="Which processed history to exclude for existing users when seen items are filtered.",
    )
    parser.add_argument(
        "--cold-start-reg",
        type=float,
        default=None,
        help="Ridge regularization for cold-start latent user vector. Defaults to ALS reg_param from meta.json.",
    )
    parser.add_argument(
        "--cold-start-min-ratings",
        type=int,
        default=3,
        help="Print a warning if fewer than this many ratings are provided for cold-start.",
    )
    args = parser.parse_args()

    if args.ratings_file is not None:
        if args.new_user_ratings is not None:
            raise SystemExit("Use either --ratings OR --ratings-file, not both.")
        args.new_user_ratings = _read_ratings_file(pathlib.Path(args.ratings_file))

    cold_start_mode = args.new_user_ratings is not None
    existing_user_mode = (args.user_id is not None) or (args.user_index is not None)
    if cold_start_mode and existing_user_mode:
        raise SystemExit("Use either --ratings/--ratings-file OR --user-id/--user-index, not both.")
    if not cold_start_mode and not existing_user_mode:
        raise SystemExit("Provide either --ratings/--ratings-file or exactly one of --user-id/--user-index.")
    if not cold_start_mode:
        if (args.user_id is None) == (args.user_index is None):
            raise SystemExit("Provide exactly one of --user-id or --user-index.")

    dataset_key = args.dataset
    processed_dir = PROCESSED_DIR / dataset_key
    model_dir = MODELS_DIR / dataset_key / "als_residual"
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Missing ALS artifacts for {dataset_key}. Expected folder: {model_dir}\n"
            f"Run: python3 scripts/collab_filtering.py --dataset {dataset_key} --stages als_final --resume --force"
        )

    mu = float(np.load(model_dir / "mu.npy"))
    b_u = np.load(model_dir / "bias_bu.npy")
    b_i = np.load(model_dir / "bias_bi.npy")
    user_factors = np.load(model_dir / "user_factors.npy")  # (num_users, rank)
    item_factors = np.load(model_dir / "item_factors.npy")  # (num_items, rank)
    meta_path = model_dir / "meta.json"
    meta: dict[str, Any] = json.load(open(meta_path, "r")) if meta_path.exists() else {}
    default_cold_start_reg = float(meta.get("reg_param", 0.1))
    cold_start_reg = float(args.cold_start_reg) if args.cold_start_reg is not None else default_cold_start_reg

    num_users = user_factors.shape[0]
    num_items = item_factors.shape[0]

    if cold_start_mode:
        parsed = _parse_movie_ratings(args.new_user_ratings)
        if len(parsed) == 0:
            raise SystemExit("--new-user-ratings/--ratings parsed to 0 pairs.")
        movie_map = _load_movie_index_map(processed_dir)
        movie_id_to_index = {
            int(mid): int(mi)
            for mid, mi in zip(
                movie_map["movieId_raw"].to_numpy(dtype=np.int64, copy=False),
                movie_map["movie_index"].to_numpy(dtype=np.int64, copy=False),
            )
        }

        # Map raw MovieLens movieIds to contiguous indices used by the ALS artifacts.
        mapped_movie_indices: list[int] = []
        mapped_ratings: list[float] = []
        exclude_movie_indices: set[int] = set()
        seen_movie_indices: set[int] = set()
        unknown_movie_ids: list[int] = []
        for movie_id_raw, rating in parsed:
            if movie_id_raw not in movie_id_to_index:
                print(f"[warn] Unknown movieId_raw={movie_id_raw} for dataset={dataset_key}; skipping.", file=sys.stderr)
                unknown_movie_ids.append(movie_id_raw)
                continue
            mi = movie_id_to_index[movie_id_raw]
            # If the same movie appears multiple times, keep the last occurrence
            # (by overwriting in the lists via a simple index map).
            if mi in seen_movie_indices:
                # Overwrite last occurrence in mapped lists.
                last_pos = mapped_movie_indices.index(mi)
                mapped_ratings[last_pos] = float(rating)
                exclude_movie_indices.add(mi)
                continue
            mapped_movie_indices.append(mi)
            mapped_ratings.append(float(rating))
            exclude_movie_indices.add(mi)
            seen_movie_indices.add(mi)

        if len(mapped_movie_indices) == 0:
            raise SystemExit("All --new-user-ratings movies were unknown to this dataset.")
        if args.cold_start_min_ratings is not None and len(mapped_movie_indices) < int(args.cold_start_min_ratings):
            print(
                f"[warn] Cold-start has only {len(mapped_movie_indices)} mapped ratings; recommendations may be noisy.",
                file=sys.stderr,
            )

        mi_arr = np.asarray(mapped_movie_indices, dtype=np.int32)
        r_arr = np.asarray(mapped_ratings, dtype=np.float32)
        b_u_new, u_new = _estimate_new_user_from_ratings(
            movie_indices=mi_arr,
            ratings=r_arr,
            mu=mu,
            bias_bi=b_i.astype(np.float32, copy=False),
            item_factors=item_factors.astype(np.float32, copy=False),
            cold_start_reg=cold_start_reg,
        )

        # Compute ALS residual score for all items for this inferred user.
        residual_scores = u_new.astype(np.float32, copy=False) @ item_factors.T.astype(np.float32, copy=False)  # (num_items,)
        pred = mu + float(b_u_new) + b_i.astype(np.float32) + residual_scores

        # Exclude items rated in the provided cold-start input.
        if args.exclude_rated and exclude_movie_indices:
            pred[list(exclude_movie_indices)] = -np.inf

    else:
        if args.user_index is not None:
            user_index = int(args.user_index)
            if user_index < 0 or user_index >= num_users:
                raise ValueError(f"--user-index out of range: {user_index} (num_users={num_users})")
        else:
            user_map = _load_user_index_map(processed_dir)
            row = user_map.loc[user_map["userId_raw"] == int(args.user_id)]
            if row.empty:
                raise ValueError(f"Unknown --user-id={args.user_id} for dataset={dataset_key}")
            user_index = int(row.iloc[0]["user_index"])

        # Compute ALS residual score for all items for this user.
        # residual_pred = U[u] dot V[item]
        residual_scores = user_factors[user_index].astype(np.float32) @ item_factors.T.astype(np.float32)  # (num_items,)
        pred = mu + float(b_u[user_index]) + b_i.astype(np.float32) + residual_scores

        # Exclude items the user already rated.
        if args.exclude_rated:
            history_name = "train.csv.gz" if args.exclude_seen_scope == "train" else "ratings_clean.csv.gz"
            rated = _load_rated_movies_for_user(processed_dir / history_name, user_index)
            if rated:
                pred[list(rated)] = -np.inf

    # Top-k by (score desc, movie_index desc) for deterministic ties.
    k = min(int(args.top_k), num_items)
    cand_idx = np.argpartition(pred, -k)[-k:]
    cand_scores = pred[cand_idx]
    order = np.lexsort((-cand_idx, -cand_scores))
    top_idx = cand_idx[order]

    titles, genres = _load_movie_titles(dataset_key, processed_dir)

    movie_map = None
    if args.format == "json" or args.show_movie_index:
        # movie_map lets us include stable raw MovieLens ids in machine output.
        movie_map_df = pd.read_csv(processed_dir / "movie_map.csv")
        movie_map = movie_map_df.set_index("movie_index")

    score_decimals = int(args.score_decimals)
    score_fmt = f"{{:.{score_decimals}f}}"

    if args.format == "json":
        recommendations: list[dict[str, Any]] = []
        for rank, mi in enumerate(top_idx[:k], start=1):
            mi = int(mi)
            title = titles[mi] if mi < len(titles) else ""
            genre = genres[mi] if mi < len(genres) else ""
            score = float(pred[mi])

            rec: dict[str, Any] = {"rank": rank, "movie_index": mi, "score": score, "title": title, "genres": genre}
            if movie_map is not None and mi in movie_map.index:
                rec["movieId_raw"] = int(movie_map.loc[mi, "movieId_raw"])
            if args.show_movie_index:
                rec["movie_index"] = mi
            recommendations.append(rec)

        payload: dict[str, Any] = {
            "dataset": dataset_key,
            "mode": "cold_start" if cold_start_mode else "existing_user",
            "top_k": k,
            "recommendations": recommendations,
        }
        if cold_start_mode:
            payload["cold_start_provided_pairs"] = len(parsed)
        else:
            payload["userId_raw"] = args.user_id
        print(json.dumps(payload, indent=2))
        return

    # Default: compact human-readable text.
    if cold_start_mode:
        # `parsed` + `unknown_movie_ids` are only defined in this branch.
        header_extra = f"from {len(mapped_movie_indices)} provided ratings"
        if unknown_movie_ids:
            header_extra += f" ({len(unknown_movie_ids)} skipped unknown movieIds)"
        if args.exclude_rated:
            header_extra += " (excluding provided rated movies)"
        print(f"Top {k} recommendations for cold-start user ({dataset_key}): {header_extra}")
    else:
        header = f"Top {k} recommendations for userId={args.user_id} ({dataset_key})"
        if args.exclude_rated:
            header += ": excluding already-rated movies"
        print(header)

    for rank, mi in enumerate(top_idx[:k], start=1):
        mi = int(mi)
        title = titles[mi] if mi < len(titles) else ""
        genre = genres[mi] if mi < len(genres) else ""
        score = float(pred[mi])

        if args.max_title_chars is not None and len(title) > int(args.max_title_chars):
            title = title[: int(args.max_title_chars)].rstrip() + "..."

        genre_disp = str(genre).replace("|", ", ")
        score_disp = score_fmt.format(score)
        if args.show_movie_index:
            print(f"{rank:2d}. {title} | movie_index={mi} | score={score_disp} | genres={genre_disp}")
        else:
            print(f"{rank:2d}. {title} | score={score_disp} | genres={genre_disp}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
