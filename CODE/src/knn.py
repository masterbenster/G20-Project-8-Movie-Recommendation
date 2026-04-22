import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

class ItemKNN:
    def __init__(self, k=20):
        self.k = k

    def fit(self, train):
        self.mu        = train["rating"].mean()
        self.user_bias = train.groupby("user_idx")["rating"].mean() - self.mu
        self.item_bias = train.groupby("item_idx")["rating"].mean() - self.mu

        self.n_users = train["user_idx"].max() + 1
        self.n_items = train["item_idx"].max() + 1

        # Center ratings
        train = train.copy()
        train["rating_centered"] = (
            train["rating"]
            - self.mu
            - train["user_idx"].map(self.user_bias).fillna(0)
            - train["item_idx"].map(self.item_bias).fillna(0)
        )

        R = csr_matrix(
            (train["rating_centered"].values, (train["user_idx"].values, train["item_idx"].values)),
            shape=(self.n_users, self.n_items)
        )
        self.R = R.astype(np.float32)
        # item-item cosine similarity (items as rows)
        self.sim = cosine_similarity(R.T, dense_output=False).astype(np.float32)

    def predict(self, df):
        n_items, n_users = self.sim.shape[0], self.R.shape[0]
        preds = []
        for _, row in df.iterrows():
            u, i = int(row["user_idx"]), int(row["item_idx"])
            ub = self.user_bias.get(u, 0)
            ib = self.item_bias.get(i, 0)

            # Fall back to bias-only for unseen users or items
            if i >= n_items or u >= n_users:
                preds.append(self.mu + ub + ib)
                continue

            # Find top-k similar items that user u has rated
            sim_row    = self.sim[i].toarray().flatten()
            user_items = self.R[u].toarray().flatten()

            rated_mask = user_items != 0
            sim_row[i] = 0  # exclude self

            if rated_mask.sum() == 0:
                preds.append(self.mu + ub + ib)
                continue

            sims   = sim_row * rated_mask
            top_k  = np.argsort(sims)[-self.k:]
            w      = sims[top_k]
            r      = user_items[top_k]

            if w.sum() == 0:
                preds.append(self.mu + ub + ib)
            else:
                preds.append(self.mu + ub + ib + np.dot(w, r) / w.sum())

        return np.array(preds).clip(0.5, 5.0)

    def explain(self, user_idx, recommended_item_idxs, train, movies, top_n=20):
        idx_to_movieid = train.drop_duplicates("item_idx").set_index("item_idx")["movieId"].to_dict()
        item_to_title  = movies.drop_duplicates("movieId").set_index("movieId")["title"].to_dict()

        user_ratings   = (train[train["user_idx"] == user_idx]
                          .sort_values("rating", ascending=False))
        actual_ratings = {int(r["item_idx"]): r["rating"]
                          for _, r in user_ratings.iterrows()}
        user_items     = self.R[user_idx].toarray().flatten()

        if user_items.sum() == 0:
            return []

        # Aggregate item-item similarity across all recommended items
        agg = np.zeros(len(user_items), dtype=np.float32)
        for item_idx in recommended_item_idxs:
            sim_row = self.sim[item_idx].toarray().flatten()
            sim_row[item_idx] = 0
            agg += sim_row * (user_items != 0)

        top_j  = np.argsort(agg)[::-1][:top_n]
        result = []
        for j in top_j:
            if agg[j] <= 0:
                break
            movie_id = idx_to_movieid.get(int(j))
            title    = item_to_title.get(movie_id, f"MovieID {movie_id}")
            result.append((title, actual_ratings.get(int(j), 0.0)))
        return result