import numpy as np

# --- Rating Prediction Metrics ---

def rmse(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean())

def mae(y_true, y_pred):
    return np.abs(y_true - y_pred).mean()

def evaluate_rating(model, df):
    preds = model.predict(df)
    y     = df["rating"].values
    return {"RMSE": round(rmse(y, preds), 4),
            "MAE":  round(mae(y, preds), 4)}


# --- Ranking Metrics ---

def precision_at_k(recommended, relevant, k):
    return len(set(recommended[:k]) & set(relevant)) / k

def recall_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0
    return len(set(recommended[:k]) & set(relevant)) / len(relevant)

def ndcg_at_k(recommended, relevant, k):
    relevant = set(relevant)
    dcg  = sum(1 / np.log2(i + 2) for i, r in enumerate(recommended[:k]) if r in relevant)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_ranking(model, train, test, movies_df, k_list=[5, 10, 20], threshold=4.0, max_users=500, n_negatives=99):
    """
    For each user: rank 1 positive test item against n_negatives random unseen items.
    Standard sampled evaluation protocol.
    """
    results = {k: {"P": [], "R": [], "NDCG": []} for k in k_list}
    max_k   = max(k_list)

    seen = train.groupby("user_idx")["item_idx"].apply(set).to_dict()

    # One positive per user: the highest-rated test item >= threshold (leave-one-out protocol)
    qualifying = test[test["rating"] >= threshold].copy()
    qualifying = (qualifying.sort_values("rating", ascending=False)
                             .drop_duplicates("user_idx"))
    positive_item = qualifying.set_index("user_idx")["item_idx"].to_dict()

    all_items = np.arange(train["item_idx"].max() + 1)
    user_list = list(positive_item.keys())[:max_users]
    rng       = np.random.default_rng(42)

    import pandas as pd
    for user_idx in user_list:
        relevant  = [positive_item[user_idx]]
        user_seen = seen.get(user_idx, set()) | set(relevant)
        unseen    = np.array([i for i in all_items if i not in user_seen])

        if len(unseen) < n_negatives:
            continue

        negatives  = rng.choice(unseen, size=n_negatives, replace=False)
        candidates = np.concatenate([relevant, negatives])
        rng.shuffle(candidates)

        df_cand = pd.DataFrame({"user_idx": user_idx, "item_idx": candidates})
        scores  = model.predict(df_cand)
        ranked  = candidates[np.argsort(scores)[::-1]]

        for k in k_list:
            results[k]["P"].append(precision_at_k(ranked, relevant, k))
            results[k]["R"].append(recall_at_k(ranked, relevant, k))
            results[k]["NDCG"].append(ndcg_at_k(ranked, relevant, k))

    summary = {}
    for k in k_list:
        summary[f"P@{k}"]    = round(np.mean(results[k]["P"]), 4)
        summary[f"R@{k}"]    = round(np.mean(results[k]["R"]), 4)
        summary[f"NDCG@{k}"] = round(np.mean(results[k]["NDCG"]), 4)
    return summary