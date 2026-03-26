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

def evaluate_ranking(model, train, test, movies_df, k_list=[5, 10, 20], threshold=4.0):
    """
    For each user in test: recommend top-max(k_list) unseen movies,
    treat test items rated >= threshold as relevant.
    """
    results = {k: {"P": [], "R": [], "NDCG": []} for k in k_list}
    max_k   = max(k_list)

    # Items each user has already seen in train
    seen = train.groupby("user_idx")["item_idx"].apply(set).to_dict()

    # Relevant items per user in test (rated >= threshold)
    relevant_items = (
        test[test["rating"] >= threshold]
        .groupby("user_idx")["item_idx"].apply(list).to_dict()
    )

    all_items = np.arange(train["item_idx"].max() + 1)

    for user_idx, relevant in relevant_items.items():
        user_seen = seen.get(user_idx, set())
        candidates = np.array([i for i in all_items if i not in user_seen])

        if len(candidates) == 0:
            continue

        # Score all unseen items
        import pandas as pd
        df_cand = pd.DataFrame({
            "user_idx": user_idx,
            "item_idx": candidates
        })
        scores = model.predict(df_cand)
        top_k_idx = candidates[np.argsort(scores)[::-1][:max_k]]

        for k in k_list:
            results[k]["P"].append(precision_at_k(top_k_idx, relevant, k))
            results[k]["R"].append(recall_at_k(top_k_idx, relevant, k))
            results[k]["NDCG"].append(ndcg_at_k(top_k_idx, relevant, k))

    # Average across users
    summary = {}
    for k in k_list:
        summary[f"P@{k}"]    = round(np.mean(results[k]["P"]), 4)
        summary[f"R@{k}"]    = round(np.mean(results[k]["R"]), 4)
        summary[f"NDCG@{k}"] = round(np.mean(results[k]["NDCG"]), 4)
    return summary