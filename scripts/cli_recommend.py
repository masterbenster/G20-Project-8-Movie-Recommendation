import argparse
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

    movies_df = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        header=None,
        names=["movieId_raw", "title", "genres"],
        dtype={"movieId_raw": np.int32, "title": str, "genres": str},
        encoding="latin-1",
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


def _load_rated_movies_for_user(train_csv_gz: pathlib.Path, user_index: int, chunksize: int = 200000) -> set[int]:
    rated: set[int] = set()
    # Stream through train.csv.gz and collect movie_index rows for this user_index.
    # This avoids loading the full dataset into RAM.
    dtypes = {"user_index": np.int32, "movie_index": np.int32}
    for chunk in pd.read_csv(train_csv_gz, compression="gzip", usecols=["user_index", "movie_index"], dtype=dtypes, chunksize=chunksize):
        sub = chunk[chunk["user_index"] == user_index]
        if len(sub) > 0:
            rated.update(sub["movie_index"].astype(int).tolist())
    return rated


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-N movie recommendations CLI (ALS residual factors).")
    parser.add_argument("--dataset", choices=["1m", "10m"], required=True)
    parser.add_argument("--user-id", type=int, default=None, help="MovieLens raw userId.")
    parser.add_argument("--user-index", type=int, default=None, help="Contiguous user_index used by processed data.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--exclude-rated", action="store_true", default=True, help="Exclude items in training history for the user.")
    args = parser.parse_args()

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

    num_users = user_factors.shape[0]
    num_items = item_factors.shape[0]

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

    # Exclude items the user already rated in training.
    if args.exclude_rated:
        train_csv_gz = processed_dir / "train.csv.gz"
        rated = _load_rated_movies_for_user(train_csv_gz, user_index)
        if rated:
            pred[list(rated)] = -np.inf

    # Top-k by (score desc, movie_index desc) for deterministic ties.
    k = min(int(args.top_k), num_items)
    cand_idx = np.argpartition(pred, -k)[-k:]
    cand_scores = pred[cand_idx]
    order = np.lexsort((-cand_idx, -cand_scores))
    top_idx = cand_idx[order]

    titles, genres = _load_movie_titles(dataset_key, processed_dir)

    for rank, mi in enumerate(top_idx[:k], start=1):
        mi = int(mi)
        title = titles[mi] if mi < len(titles) else ""
        genre = genres[mi] if mi < len(genres) else ""
        score = float(pred[mi])
        print(f"{rank}. movie_index={mi} score={score:.4f} title={title} genres={genre}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

