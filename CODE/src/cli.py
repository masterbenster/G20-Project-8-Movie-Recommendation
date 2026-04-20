import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import pickle
import pandas as pd
import numpy as np
from src.data import load_1m, load_splits, save_splits, time_split

def get_recommendations(model, user_idx, train, movies, n=10, all_ratings=None):
    # Exclude everything the user has ever rated, not just their training items
    ratings_ref = all_ratings if all_ratings is not None else train
    seen = set(ratings_ref[ratings_ref["user_idx"] == user_idx]["item_idx"].values)

    all_items = np.arange(train["item_idx"].max() + 1)
    candidates = np.array([i for i in all_items if i not in seen])

    df_cand = pd.DataFrame({"user_idx": user_idx, "item_idx": candidates})
    scores  = model.predict(df_cand)
    top_idx = candidates[np.argsort(scores)[::-1][:n]]

    item_to_movieid = train.drop_duplicates("item_idx").set_index("item_idx")["movieId"].to_dict()

    results = []
    for idx, score in zip(top_idx, np.sort(scores)[::-1][:n]):
        movie_id = item_to_movieid.get(idx)
        if movie_id is None:
            continue
        title = movies[movies["movieId"] == movie_id]["title"].values
        title = title[0] if len(title) > 0 else f"MovieID {movie_id}"
        results.append((title, round(float(score), 3)))
    return results


def main():
    parser = argparse.ArgumentParser(description="Movie Recommender CLI")
    parser.add_argument("--user",  type=int, required=True, help="User ID (1-6040 for ml-1m)")
    parser.add_argument("--model", type=str, default="als", choices=["als","knn","neumf"], help="Model to use")
    parser.add_argument("--topn",  type=int, default=10, help="Number of recommendations")
    args = parser.parse_args()

    import joblib
    data_dir    = os.path.join(os.path.dirname(__file__), "..", "..", "DATA")
    weights_dir = os.path.join(os.path.dirname(__file__), "..", "weights")
    dataset_dir = os.path.join(data_dir, "ml-1m")

    weight_path = os.path.join(weights_dir, f"{args.model}.joblib")
    if not os.path.exists(weight_path):
        print(f"No saved weights found at {weight_path}. Run CODE/train.py first.")
        return

    print(f"Loading data and model '{args.model}'...")
    ratings, movies, users = load_1m(path=dataset_dir)

    cached = load_splits(dataset_dir)
    if cached:
        train, val, test = cached
    else:
        train, val, test = time_split(ratings)
        save_splits(train, val, test, dataset_dir)

    model = joblib.load(weight_path)
    print(f"  Loaded weights from {os.path.relpath(os.path.abspath(weight_path))}")

    # Map public userId to internal user_idx
    user_map = train.drop_duplicates("userId").set_index("userId")["user_idx"].to_dict()
    if args.user not in user_map:
        print(f"User {args.user} not found. Valid range: {min(user_map)} - {max(user_map)}")
        return

    user_idx = user_map[args.user]
    recs = get_recommendations(model, user_idx, train, movies, n=args.topn, all_ratings=ratings)

    explanations = None
    if hasattr(model, "explain"):
        title_to_idx = (train.drop_duplicates("item_idx")
                             .merge(movies, on="movieId")[["title", "item_idx"]]
                             .set_index("title")["item_idx"].to_dict())
        rec_item_idxs = [title_to_idx[title] for title, _ in recs if title in title_to_idx]
        if len(rec_item_idxs) == len(recs):
            explanations = model.explain(user_idx, rec_item_idxs, train, movies)

    print(f"\nTop {args.topn} recommendations for User {args.user} ({args.model.upper()}):")
    for i, (title, score) in enumerate(recs, 1):
        print(f"  {i:2}. {title}  (score: {score})")

    if explanations:
        print(f"\nWhy these recommendations ({args.model.upper()}):")
        for i, ((title, _), reasons) in enumerate(zip(recs, explanations), 1):
            print(f"\n  {i:2}. {title}")
            for reason_title, rating in reasons:
                print(f"        - {reason_title} ({rating:.1f}★)")


if __name__ == "__main__":
    main()