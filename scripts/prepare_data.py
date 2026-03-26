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
    # Time-ordered per-user 80/10/10 split.
    # We keep the oldest ratings in train, the next block in validation,
    # and the most recent ratings in test.
    counts = ratings.groupby("userId")["userId"].transform("size").astype(np.int32)
    ratings = ratings[counts >= 3].copy()
    ratings = ratings.sort_values(["userId", "timestamp", "movieId"]).copy()

    counts = ratings.groupby("userId")["userId"].transform("size").astype(np.int32)
    count_arr = counts.to_numpy(dtype=np.int32, copy=False)
    val_count = np.maximum(1, np.floor(count_arr * 0.1).astype(np.int32))
    test_count = np.maximum(1, np.floor(count_arr * 0.1).astype(np.int32))
    train_count = count_arr - val_count - test_count
    if np.any(train_count < 1):
        raise ValueError("Time-aware 80/10/10 split produced a user with empty train set.")

    ratings["rank"] = ratings.groupby("userId").cumcount().astype(np.int32)
    ratings["train_count"] = train_count
    ratings["val_count"] = val_count

    train_mask = ratings["rank"] < ratings["train_count"]
    val_mask = (ratings["rank"] >= ratings["train_count"]) & (
        ratings["rank"] < (ratings["train_count"] + ratings["val_count"])
    )
    test_mask = ~(train_mask | val_mask)

    train_df = ratings[train_mask].copy()
    val_df = ratings[val_mask].copy()
    test_df = ratings[test_mask].copy()

    # Keep split-specific columns tidy.
    drop_cols = ["rank", "train_count", "val_count"]
    train_df = train_df.drop(columns=drop_cols)
    val_df = val_df.drop(columns=drop_cols)
    test_df = test_df.drop(columns=drop_cols)
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


def prepare_dataset(spec: DatasetSpec, num_negatives: int, seed: int) -> None:
    # Backward-compatible wrapper: original behavior excluded all previously-seen items.
    prepare_dataset_with_negative_scope(spec, num_negatives=num_negatives, seed=seed, negative_scope="all")


def _validate_split_outputs(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ratings_clean_out: pd.DataFrame,
    val_negs_df: pd.DataFrame | None = None,
    test_negs_df: pd.DataFrame | None = None,
    num_negatives: int | None = None,
) -> dict[str, int]:
    ratings_pairs = ratings_clean_out[["user_index", "movie_index"]].drop_duplicates()

    def _overlap_count(left: pd.DataFrame, right: pd.DataFrame) -> int:
        merged = left[["user_index", "movie_index"]].merge(
            right[["user_index", "movie_index"]],
            on=["user_index", "movie_index"],
            how="inner",
        )
        return int(len(merged))

    def _negative_summary(label: str, negs_df: pd.DataFrame) -> dict[str, int]:
        event_sizes = negs_df.groupby(["user_index", "pos_movie_index"]).size()
        if event_sizes.empty:
            raise ValueError(f"{label}: no ranking candidate groups were generated.")
        if num_negatives is not None and not bool((event_sizes == int(num_negatives)).all()):
            raise ValueError(f"{label}: expected {num_negatives} negatives per event, found inconsistent counts.")

        seen_overlap = (
            negs_df[["user_index", "neg_movie_index"]]
            .rename(columns={"neg_movie_index": "movie_index"})
            .merge(ratings_pairs, on=["user_index", "movie_index"], how="inner")
        )
        if len(seen_overlap) > 0:
            raise ValueError(f"{label}: generated negatives overlap with known user history ({len(seen_overlap)} rows).")

        return {
            f"{label}_events": int(event_sizes.shape[0]),
            f"{label}_neg_rows_min": int(event_sizes.min()),
            f"{label}_neg_rows_max": int(event_sizes.max()),
            f"{label}_seen_overlap_rows": int(len(seen_overlap)),
        }

    train_val_overlap = _overlap_count(train_df, val_df)
    train_test_overlap = _overlap_count(train_df, test_df)
    val_test_overlap = _overlap_count(val_df, test_df)
    if train_val_overlap or train_test_overlap or val_test_overlap:
        raise ValueError("Train/val/test splits overlap.")

    summary = {
        "train_val_overlap_rows": train_val_overlap,
        "train_test_overlap_rows": train_test_overlap,
        "val_test_overlap_rows": val_test_overlap,
    }
    if val_negs_df is not None and test_negs_df is not None:
        summary.update(_negative_summary("val", val_negs_df))
        summary.update(_negative_summary("test", test_negs_df))
    return summary


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

    # Write artifacts.
    _write_gz_csv(ratings_clean_out, out_dir / "ratings_clean.csv.gz")
    _write_gz_csv(train_df, out_dir / "train.csv.gz")
    _write_gz_csv(val_df, out_dir / "val.csv.gz")
    _write_gz_csv(test_df, out_dir / "test.csv.gz")
    for stale_name in ("val_negs.csv.gz", "test_negs.csv.gz"):
        stale_path = out_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    user_map.to_csv(out_dir / "user_map.csv", index=False)
    movie_map.to_csv(out_dir / "movie_map.csv", index=False)

    validation_summary = _validate_split_outputs(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        ratings_clean_out=ratings_clean_out,
    )

    meta = {
        "dataset": spec.key,
        "seed": seed,
        "num_negatives": num_negatives,
        "negative_scope": negative_scope,
        "num_users": num_users,
        "num_movies": num_movies,
        "num_ratings_clean": int(len(ratings_clean_out)),
        "num_train": int(len(train_df)),
        "num_val": int(len(val_df)),
        "num_test": int(len(test_df)),
        "num_val_events": int(len(val_df)),
        "num_test_events": int(len(test_df)),
        "ranking_eval": {
            "negative_scope": negative_scope,
            "num_negatives": int(num_negatives),
            "materialized_candidate_files": False,
            "default_max_events": {"1m": 20000, "10m": 20000},
        },
        "split_policy": {
            "type": "time_aware_user_ratio",
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "min_user_ratings": 3,
            "ordering": "oldest_to_newest",
        },
        "validation": validation_summary,
        "note": (
            "Time-aware per-user 80/10/10 split after dedup by latest (userId,movieId). "
            "Ranking negatives are generated on the fly during evaluation."
        ),
    }
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
        help="Recorded ranking-eval policy. Candidate negatives are generated on the fly during evaluation.",
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
        if args.skip_existing and meta_path.exists():
            print(f"[skip] {spec.key}: {meta_path} already exists")
            continue

        print(f"[prep] {spec.key}: reading raw ratings and building corrected splits...")
        prepare_dataset_with_negative_scope(
            spec,
            num_negatives=args.num_negatives,
            seed=args.seed,
            negative_scope=args.negative_scope,
        )
        print(f"[done] {spec.key}: outputs written to {out_dir}")


if __name__ == "__main__":
    main()
