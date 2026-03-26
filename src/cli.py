import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import pickle
import pandas as pd
import numpy as np
from src.data import load_1m

def get_recommendations(model, user_idx, train, movies, n=10):
    seen = set(train[train["user_idx"] == user_idx]["item_idx"].values)
    all_items = np.arange(train["item_idx"].max() + 1)
    candidates = np.array([i for i in all_items if i not in seen])

    df_cand = pd.DataFrame({"user_idx": user_idx, "item_idx": candidates})
    scores  = model.predict(df_cand)
    top_idx = candidates[np.argsort(scores)[::-1][:n]]

    # Map item_idx back to movieId
    idx_to_movie = dict(zip(
        train["item_idx"].astype("category").cat.codes if False else
        pd.Categorical(train["movieId"]).codes,
        train["movieId"]
    ))
    # Simpler: build from train directly
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

    print(f"Loading data and model '{args.model}'...")
    ratings, movies, users = load_1m()

    from src.data import time_split
    train, val, test = time_split(ratings)

    if args.model == "als":
        from src.als import ALSModel
        model = ALSModel(factors=50, iterations=20, regularization=0.1)
        model.fit(train)
    elif args.model == "knn":
        from src.knn import ItemKNN
        model = ItemKNN(k=20)
        model.fit(train)
    elif args.model == "neumf":
        from src.neumf import NeuMFModel
        model = NeuMFModel(epochs=10)
        model.fit(train)

    # Map public userId to internal user_idx
    user_map = train.drop_duplicates("userId").set_index("userId")["user_idx"].to_dict()
    if args.user not in user_map:
        print(f"User {args.user} not found. Valid range: {min(user_map)} - {max(user_map)}")
        return

    user_idx = user_map[args.user]
    recs = get_recommendations(model, user_idx, train, movies, n=args.topn)

    print(f"\nTop {args.topn} recommendations for User {args.user} ({args.model.upper()}):")
    for i, (title, score) in enumerate(recs, 1):
        print(f"  {i:2}. {title}  (score: {score})")


if __name__ == "__main__":
    main()