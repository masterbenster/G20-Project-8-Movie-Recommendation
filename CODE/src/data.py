import os
import pandas as pd

def load_splits(dataset_dir):
    """Load pre-saved train/val/test splits from dataset_dir, or return None if missing."""
    paths = {s: os.path.join(dataset_dir, f"{s}.parquet") for s in ("train", "val", "test")}
    if all(os.path.exists(p) for p in paths.values()):
        print(f"  Loading cached splits from {os.path.relpath(os.path.abspath(dataset_dir))}")
        return (pd.read_parquet(paths["train"]),
                pd.read_parquet(paths["val"]),
                pd.read_parquet(paths["test"]))
    return None

def save_splits(train, val, test, dataset_dir):
    """Save train/val/test splits as parquet files inside dataset_dir."""
    for name, df in (("train", train), ("val", val), ("test", test)):
        df.to_parquet(os.path.join(dataset_dir, f"{name}.parquet"), index=False)
    print(f"  Saved splits to {dataset_dir}\n")

def load_1m(path="ml-1m"):
    ratings = pd.read_csv(f"{path}/ratings.dat", sep="::", engine="python",
                          names=["userId","movieId","rating","timestamp"])
    movies  = pd.read_csv(f"{path}/movies.dat", sep="::", engine="python",
                          encoding="latin-1",
                          names=["movieId","title","genres"])
    users   = pd.read_csv(f"{path}/users.dat", sep="::", engine="python",
                          names=["userId","gender","age","occupation","zip"])

    ratings["user_idx"] = ratings["userId"].astype("category").cat.codes
    ratings["item_idx"] = ratings["movieId"].astype("category").cat.codes
    return ratings, movies, users

def load_10m(path="ml-10M100K"):
    ratings = pd.read_csv(f"{path}/ratings.dat", sep="::", engine="python",
                          names=["userId","movieId","rating","timestamp"])
    movies  = pd.read_csv(f"{path}/movies.dat", sep="::", engine="python",
                          encoding="utf-8",
                          names=["movieId","title","genres"])
    tags    = pd.read_csv(f"{path}/tags.dat", sep="::", engine="python",
                          encoding="utf-8",
                          names=["userId","movieId","tag","timestamp"])

    ratings["user_idx"] = ratings["userId"].astype("category").cat.codes
    ratings["item_idx"] = ratings["movieId"].astype("category").cat.codes
    return ratings, movies, tags

def time_split(df, val_ratio=0.1, test_ratio=0.1, min_train=3):
    df = df.sort_values(["userId","timestamp"]).reset_index(drop=True)

    train_parts, val_parts, test_parts = [], [], []

    for _, group in df.groupby("userId"):
        n = len(group)

        # Not enough ratings for a time split, put all in train
        if n < min_train + 2:
            train_parts.append(group)
            continue

        n_test = max(1, round(n * test_ratio))
        n_val = max(1, round(n * val_ratio))

        # ensure training set never shrinks below min_train
        n_val = min(n_val, n - n_test - min_train)
        n_test = min(n_test, n - n_val - min_train)

        train_parts.append(group.iloc[:n - n_val - n_test])
        val_parts.append(group.iloc[n - n_val - n_test:n - n_test])
        test_parts.append(group.iloc[n - n_test:])

    train = pd.concat(train_parts).drop(columns=["rank"], errors="ignore")
    val   = pd.concat(val_parts).drop(columns=["rank"], errors="ignore")
    test  = pd.concat(test_parts).drop(columns=["rank"], errors="ignore")

    # Report acutal split sizes
    total = len(df)
    print(f"  Split: train={len(train)} ({100*len(train)/total:.1f}%)  "
        f"val={len(val)} ({100*len(val)/total:.1f}%)  "
        f"test={len(test)} ({100*len(test)/total:.1f}%)")

    return train, val, test