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
        self.R = R
        # item-item cosine similarity (items as rows)
        self.sim = cosine_similarity(R.T, dense_output=False)

    def predict(self, df):
        preds = []
        for _, row in df.iterrows():
            u, i = int(row["user_idx"]), int(row["item_idx"])
            ub = self.user_bias.get(u, 0)
            ib = self.item_bias.get(i, 0)

            # Find top-k similar items that user u has rated
            sim_row   = self.sim[i].toarray().flatten()
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