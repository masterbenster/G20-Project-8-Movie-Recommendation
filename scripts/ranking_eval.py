import pathlib

import numpy as np
import pandas as pd


def read_csv_gz(path: pathlib.Path, *, usecols=None, dtypes=None) -> pd.DataFrame:
    if dtypes is None:
        return pd.read_csv(path, compression="gzip", usecols=usecols)
    return pd.read_csv(path, compression="gzip", usecols=usecols, dtype=dtypes)


def default_max_ranking_events(dataset_key: str) -> int | None:
    if dataset_key in {"1m", "10m"}:
        return 20_000
    return 20_000


def sample_split_events(split_df: pd.DataFrame, *, max_events: int | None, seed: int) -> tuple[pd.DataFrame, dict[str, int | None]]:
    events = split_df[["user_index", "movie_index"]].copy().reset_index(drop=True)
    total_events = int(len(events))
    sampled_all = max_events is None or total_events <= int(max_events)

    if not sampled_all:
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(total_events, size=int(max_events), replace=False))
        events = events.iloc[chosen].reset_index(drop=True)

    events = events.rename(columns={"movie_index": "pos_movie_index"})
    events.insert(0, "event_id", np.arange(len(events), dtype=np.int64))

    return events, {
        "events_total": total_events,
        "events_evaluated": int(len(events)),
        "max_events": None if max_events is None else int(max_events),
    }


def build_candidate_groups(
    *,
    split_df: pd.DataFrame,
    rated_source_df: pd.DataFrame,
    num_movies: int,
    num_negatives: int,
    seed: int,
    max_events: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | None]]:
    sampled_events, summary = sample_split_events(split_df, max_events=max_events, seed=seed)
    sampled_users = sampled_events["user_index"].drop_duplicates().to_numpy(dtype=np.int32, copy=False)

    rated_subset = rated_source_df[rated_source_df["user_index"].isin(sampled_users)][["user_index", "movie_index"]].copy()
    rated_by_user: dict[int, set[int]] = {}
    for user_index, movie_index in zip(
        rated_subset["user_index"].to_numpy(dtype=np.int32, copy=False),
        rated_subset["movie_index"].to_numpy(dtype=np.int32, copy=False),
    ):
        rated_by_user.setdefault(int(user_index), set()).add(int(movie_index))

    rng = np.random.default_rng(seed + 17)
    rows: list[tuple[int, int, int, int]] = []
    for event_id, user_index, pos_movie_index in sampled_events.itertuples(index=False):
        rated_set = rated_by_user.get(int(user_index), set())
        negs: set[int] = set()
        while len(negs) < num_negatives:
            remaining = num_negatives - len(negs)
            batch = rng.integers(0, num_movies, size=remaining * 3, dtype=np.int32)
            for movie_index in batch:
                movie_index_int = int(movie_index)
                if movie_index_int == int(pos_movie_index) or movie_index_int in rated_set:
                    continue
                negs.add(movie_index_int)
                if len(negs) >= num_negatives:
                    break
        for neg_movie_index in negs:
            rows.append((int(event_id), int(user_index), int(pos_movie_index), int(neg_movie_index)))

    negs_df = pd.DataFrame(rows, columns=["event_id", "user_index", "pos_movie_index", "neg_movie_index"])
    summary["negative_rows"] = int(len(negs_df))
    summary["num_negatives"] = int(num_negatives)
    return sampled_events, negs_df, summary
