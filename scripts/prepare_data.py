import argparse
import gzip
import json
import os
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


@dataclass(frozen=True)
class DatasetSpec:
    key: str  # "1m" or "10m"
    zip_path: pathlib.Path
    extract_dir: pathlib.Path
    ratings_relpath: str
    movies_relpath: str | None = None


def _ensure_unzipped(spec: DatasetSpec) -> None:
    if spec.extract_dir.exists() and any(spec.extract_dir.iterdir()):
        return

    import zipfile

    spec.extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(spec.zip_path, "r") as zf:
        # Extract into spec.extract_dir by recreating the first path segment stripping.
        # The zips contain a top-level folder like "ml-1m/".
        for member in zf.infolist():
            parts = member.filename.split("/", 1)
            if len(parts) != 2:
                continue
            rel = parts[1]
            if not rel:
                continue
            target = spec.extract_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if member.is_dir():
                continue
            with zf.open(member, "r") as src, open(target, "wb") as dst:
                dst.write(src.read())


def _read_ratings(spec: DatasetSpec) -> pd.DataFrame:
    ratings_path = spec.extract_dir / spec.ratings_relpath

    # MovieLens 1M/10M both use "::" separators in ratings.dat.
    # 1M users/movies are int; rating is in [0.5, 5.0]; timestamp is int seconds.
    df = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        header=None,
        names=["userId", "movieId", "rating", "timestamp"],
        dtype={
            "userId": np.int32,
            "movieId": np.int32,
            "rating": np.float32,
            "timestamp": np.int64,
        },
    )
    return df


def _dedup_latest_by_user_movie(ratings: pd.DataFrame) -> pd.DataFrame:
    # If a user rated the same movie multiple times, keep the most recent rating.
    ratings = ratings.sort_values(["userId", "movieId", "timestamp"])
    ratings = ratings.drop_duplicates(["userId", "movieId"], keep="last")
    return ratings


def _time_aware_split(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # For each user:
    # - last rating -> test
    # - second last rating -> validation
    # - remaining -> train
    # Users with < 3 interactions are removed.
    counts = ratings.groupby("userId")["userId"].transform("size")
    ratings = ratings[counts >= 3].copy()
    ratings = ratings.sort_values(["userId", "timestamp"])
    ratings["rank"] = ratings.groupby("userId").cumcount()
    ratings["user_count"] = counts[counts >= 3]
    test_mask = ratings["rank"] == (ratings["user_count"] - 1)
    val_mask = ratings["rank"] == (ratings["user_count"] - 2)

    train_df = ratings[~(test_mask | val_mask)].copy()
    val_df = ratings[val_mask].copy()
    test_df = ratings[test_mask].copy()

    # Keep split-specific columns tidy.
    train_df = train_df.drop(columns=["rank", "user_count"])
    val_df = val_df.drop(columns=["rank", "user_count"])
    test_df = test_df.drop(columns=["rank", "user_count"])
    return train_df, val_df, test_df


def _build_mappings(ratings_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Factorize so indices are contiguous [0..N).
    user_codes, user_uniques = pd.factorize(ratings_clean["userId"], sort=False)
    movie_codes, movie_uniques = pd.factorize(ratings_clean["movieId"], sort=False)
    ratings_clean = ratings_clean.copy()
    ratings_clean["user_index"] = user_codes.astype(np.int32)
    ratings_clean["movie_index"] = movie_codes.astype(np.int32)

    user_map = pd.DataFrame({"userId_raw": user_uniques, "user_index": np.arange(len(user_uniques), dtype=np.int32)})
    movie_map = pd.DataFrame(
        {"movieId_raw": movie_uniques, "movie_index": np.arange(len(movie_uniques), dtype=np.int32)}
    )
    return ratings_clean, user_map, movie_map


def _write_gz_csv(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # gzip compression keeps these files small enough for lab usage.
    with gzip.open(out_path, "wt") as f:
        df.to_csv(f, index=False)


def _sample_negatives_for_split(
    split_df: pd.DataFrame,
    rated_by_user: dict[int, set[int]],
    num_movies: int,
    num_negatives: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    all_movies = np.arange(num_movies, dtype=np.int32)

    rows = []
    # Each row in split_df corresponds to one (user_index, pos_movie_index).
    for user_index, pos_movie_index in zip(split_df["user_index"].to_numpy(), split_df["movie_index"].to_numpy()):
        rated_set = rated_by_user.get(int(user_index), set())
        negs: set[int] = set()
        # Rejection sampling with batched draws.
        while len(negs) < num_negatives:
            remaining = num_negatives - len(negs)
            batch = rng.integers(0, num_movies, size=remaining * 3, dtype=np.int32)
            for m in batch:
                mi = int(m)
                if mi in rated_set:
                    continue
                if mi == int(pos_movie_index):
                    continue
                negs.add(mi)
                if len(negs) >= num_negatives:
                    break

        for neg_movie_index in negs:
            rows.append((int(user_index), int(pos_movie_index), int(neg_movie_index)))

    return pd.DataFrame(rows, columns=["user_index", "pos_movie_index", "neg_movie_index"])


def prepare_dataset(spec: DatasetSpec, num_negatives: int, seed: int) -> None:
    # Backward-compatible wrapper: original behavior excluded all previously-seen items.
    prepare_dataset_with_negative_scope(spec, num_negatives=num_negatives, seed=seed, negative_scope="all")


def _sample_negatives_with_scope(
    *,
    negative_scope: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ratings_clean_out: pd.DataFrame,
    num_movies: int,
    num_negatives: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Negative sampling pool depends on scope:
    # - "all": exclude any item the user interacted with anywhere (train+val+test)
    # - "train_only": exclude only items the user interacted with in training
    if negative_scope == "all":
        rated_source_df = ratings_clean_out
    elif negative_scope == "train_only":
        rated_source_df = train_df
    else:
        raise ValueError(f"Unknown negative_scope: {negative_scope}")

    rated_by_user: dict[int, set[int]] = {}
    for user_index, movie_index in zip(
        rated_source_df["user_index"].to_numpy(), rated_source_df["movie_index"].to_numpy()
    ):
        u = int(user_index)
        rated_by_user.setdefault(u, set()).add(int(movie_index))

    val_negs_df = _sample_negatives_for_split(
        split_df=val_df,
        rated_by_user=rated_by_user,
        num_movies=num_movies,
        num_negatives=num_negatives,
        seed=seed + 1,
    )
    test_negs_df = _sample_negatives_for_split(
        split_df=test_df,
        rated_by_user=rated_by_user,
        num_movies=num_movies,
        num_negatives=num_negatives,
        seed=seed + 2,
    )
    return val_negs_df, test_negs_df


def prepare_dataset_with_negative_scope(
    spec: DatasetSpec,
    num_negatives: int,
    seed: int,
    *,
    negative_scope: str,
) -> None:
    out_dir = PROCESSED_DIR / spec.key
    out_dir.mkdir(parents=True, exist_ok=True)

    _ensure_unzipped(spec)
    ratings_raw = _read_ratings(spec)

    # Dedup on (userId, movieId) by keeping the latest timestamp.
    ratings_clean = _dedup_latest_by_user_movie(ratings_raw)
    train_df_raw, val_df_raw, test_df_raw = _time_aware_split(ratings_clean)

    # Build consistent indices based on the full cleaned ratings.
    ratings_clean_indexed, user_map, movie_map = _build_mappings(ratings_clean)
    num_users = int(user_map["user_index"].max()) + 1
    num_movies = int(movie_map["movie_index"].max()) + 1

    # Map splits to indices (by merging on raw IDs).
    train_df = train_df_raw.merge(
        ratings_clean_indexed[["userId", "movieId", "user_index", "movie_index"]],
        on=["userId", "movieId"],
        how="left",
    )
    val_df = val_df_raw.merge(
        ratings_clean_indexed[["userId", "movieId", "user_index", "movie_index"]],
        on=["userId", "movieId"],
        how="left",
    )
    test_df = test_df_raw.merge(
        ratings_clean_indexed[["userId", "movieId", "user_index", "movie_index"]],
        on=["userId", "movieId"],
        how="left",
    )

    # Drop raw IDs; downstream models generally want contiguous indices.
    train_df = train_df[["user_index", "movie_index", "rating", "timestamp"]].copy()
    val_df = val_df[["user_index", "movie_index", "rating", "timestamp"]].copy()
    test_df = test_df[["user_index", "movie_index", "rating", "timestamp"]].copy()

    ratings_clean_out = ratings_clean_indexed[["user_index", "movie_index", "rating", "timestamp"]].copy()

    val_negs_df, test_negs_df = _sample_negatives_with_scope(
        negative_scope=negative_scope,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        ratings_clean_out=ratings_clean_out,
        num_movies=num_movies,
        num_negatives=num_negatives,
        seed=seed,
    )

    # Write artifacts.
    _write_gz_csv(ratings_clean_out, out_dir / "ratings_clean.csv.gz")
    _write_gz_csv(train_df, out_dir / "train.csv.gz")
    _write_gz_csv(val_df, out_dir / "val.csv.gz")
    _write_gz_csv(test_df, out_dir / "test.csv.gz")
    _write_gz_csv(val_negs_df, out_dir / "val_negs.csv.gz")
    _write_gz_csv(test_negs_df, out_dir / "test_negs.csv.gz")
    user_map.to_csv(out_dir / "user_map.csv", index=False)
    movie_map.to_csv(out_dir / "movie_map.csv", index=False)

    meta = {
        "dataset": spec.key,
        "seed": seed,
        "num_negatives": num_negatives,
        "num_users": num_users,
        "num_movies": num_movies,
        "num_ratings_clean": int(len(ratings_clean_out)),
        "num_train": int(len(train_df)),
        "num_val": int(len(val_df)),
        "num_test": int(len(test_df)),
        "num_val_negs_rows": int(len(val_negs_df)),
        "num_test_negs_rows": int(len(test_negs_df)),
        "note": (
            "Time split: per-user last->test, second-last->val, rest->train after dedup by latest (userId,movieId). "
            f"Negative scope: {negative_scope}."
        ),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def regenerate_negatives_only(
    spec: DatasetSpec,
    *,
    num_negatives: int,
    seed: int,
    negative_scope: str,
) -> None:
    out_dir = PROCESSED_DIR / spec.key
    meta = json.load(open(out_dir / "meta.json", "r"))
    num_users = int(meta["num_users"])
    num_movies = int(meta["num_movies"])

    processed_dir = PROCESSED_DIR / spec.key
    dtypes_train = {"user_index": np.int32, "movie_index": np.int32, "rating": np.float32, "timestamp": np.int64}
    train_df = pd.read_csv(processed_dir / "train.csv.gz", compression="gzip", dtype=dtypes_train)
    val_df = pd.read_csv(processed_dir / "val.csv.gz", compression="gzip", dtype=dtypes_train)[
        ["user_index", "movie_index", "rating", "timestamp"]
    ]
    test_df = pd.read_csv(processed_dir / "test.csv.gz", compression="gzip", dtype=dtypes_train)[
        ["user_index", "movie_index", "rating", "timestamp"]
    ]

    if negative_scope == "all":
        # Need full interaction history to exclude seen items.
        dtypes_all = {"user_index": np.int32, "movie_index": np.int32, "rating": np.float32, "timestamp": np.int64}
        ratings_clean_out = pd.read_csv(
            processed_dir / "ratings_clean.csv.gz", compression="gzip", dtype=dtypes_all
        )[
            ["user_index", "movie_index", "rating", "timestamp"]
        ]
    elif negative_scope == "train_only":
        ratings_clean_out = None
    else:
        raise ValueError(f"Unknown negative_scope: {negative_scope}")

    # Create negatives.
    val_negs_df, test_negs_df = _sample_negatives_with_scope(
        negative_scope=negative_scope,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        ratings_clean_out=ratings_clean_out if ratings_clean_out is not None else train_df,
        num_movies=num_movies,
        num_negatives=num_negatives,
        seed=seed,
    )

    _write_gz_csv(val_negs_df, out_dir / "val_negs.csv.gz")
    _write_gz_csv(test_negs_df, out_dir / "test_negs.csv.gz")

    # Update meta with new scope and row counts.
    meta["num_negatives"] = num_negatives
    meta["num_val_negs_rows"] = int(len(val_negs_df))
    meta["num_test_negs_rows"] = int(len(test_negs_df))
    meta["note"] = meta.get("note", "") + f" (negative_scope={negative_scope}, regenerated=True)"
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MovieLens data (Step 1: pipeline, splits, negatives).")
    parser.add_argument("--dataset", choices=["1m", "10m", "both"], default="both")
    parser.add_argument("--num-negatives", type=int, default=99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--negative-scope",
        choices=["all", "train_only"],
        default="all",
        help="What to exclude when sampling negatives: all user interactions vs train-only history.",
    )
    parser.add_argument(
        "--regenerate-negatives",
        action="store_true",
        help="If processed data exists, only regenerate val/test negatives with the new scope.",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip if meta.json exists for dataset.")
    args = parser.parse_args()

    zips = {
        "1m": DatasetSpec(
            key="1m",
            zip_path=pathlib.Path(__file__).resolve().parents[1] / "ml-1m.zip",
            extract_dir=RAW_DIR / "ml-1m",
            ratings_relpath="ratings.dat",
            movies_relpath="movies.dat",
        ),
        "10m": DatasetSpec(
            key="10m",
            zip_path=pathlib.Path(__file__).resolve().parents[1] / "ml-10m.zip",
            extract_dir=RAW_DIR / "ml-10M100K",
            ratings_relpath="ratings.dat",
            movies_relpath="movies.dat",
        ),
    }

    targets = ["1m", "10m"] if args.dataset == "both" else [args.dataset]
    for key in targets:
        spec = zips[key]
        out_dir = PROCESSED_DIR / spec.key
        meta_path = out_dir / "meta.json"
        if args.regenerate_negatives and meta_path.exists():
            print(f"[regen] {spec.key}: regenerating negatives only (scope={args.negative_scope})...")
            regenerate_negatives_only(
                spec,
                num_negatives=args.num_negatives,
                seed=args.seed,
                negative_scope=args.negative_scope,
            )
            print(f"[done] {spec.key}: negatives updated in {out_dir}")
            continue

        if args.skip_existing and meta_path.exists():
            print(f"[skip] {spec.key}: {meta_path} already exists")
            continue

        print(f"[prep] {spec.key}: reading raw ratings and building splits/negatives...")
        prepare_dataset_with_negative_scope(
            spec,
            num_negatives=args.num_negatives,
            seed=args.seed,
            negative_scope=args.negative_scope,
        )
        print(f"[done] {spec.key}: outputs written to {out_dir}")


if __name__ == "__main__":
    main()

