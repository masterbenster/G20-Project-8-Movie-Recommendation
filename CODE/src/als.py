import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from threadpoolctl import threadpool_limits
threadpool_limits(1, "blas")

import implicit
import numpy as np
import torch
from scipy.sparse import csr_matrix

def _als_class():
    if torch.cuda.is_available() and not torch.version.hip:
        try:
            print("  ALS: using GPU (CUDA)")
            return implicit.gpu.als.AlternatingLeastSquares
        except AttributeError:
            pass
    if torch.cuda.is_available() and torch.version.hip:
        print("ALS: ROCm detected — implicit GPU backend requires CUDA/cupy, using CPU")
    else:
        print("ALS: using CPU")
    return implicit.cpu.als.AlternatingLeastSquares

class ALSModel:
    def __init__(self, factors=50, iterations=20, regularization=0.1):
        self.model = _als_class()(
            factors=factors,
            iterations=iterations,
            regularization=regularization
        )

    def fit(self, train):
        self.mu        = train["rating"].mean()
        self.user_bias = train.groupby("user_idx")["rating"].mean() - self.mu
        self.item_bias = train.groupby("item_idx")["rating"].mean() - self.mu

        # Center ratings before factorization
        train = train.copy()
        train["rating_centered"] = (
            train["rating"]
            - self.mu
            - train["user_idx"].map(self.user_bias).fillna(0)
            - train["item_idx"].map(self.item_bias).fillna(0)
        )

        self.n_users = train["user_idx"].max() + 1
        self.n_items = train["item_idx"].max() + 1

        R = csr_matrix(
            (train["rating_centered"].values, (train["user_idx"].values, train["item_idx"].values)),
            shape=(self.n_users, self.n_items)
        )
        self.R = R
        self.model.fit(R)

    def predict(self, df):
        uf  = self.model.user_factors
        itf = self.model.item_factors

        u_idx = df["user_idx"].values.astype(int)
        i_idx = df["item_idx"].values.astype(int)

        ub = df["user_idx"].map(self.user_bias).fillna(0).values
        ib = df["item_idx"].map(self.item_bias).fillna(0).values

        # Vectorized dot product
        valid_u = u_idx < len(uf)
        valid_i = i_idx < len(itf)
        valid   = valid_u & valid_i

        dots = np.zeros(len(df))
        dots[valid] = np.sum(uf[u_idx[valid]] * itf[i_idx[valid]], axis=1)

        return (self.mu + ub + ib + dots).clip(0.5, 5.0)

    def recommend(self, user_idx, N=10):
        ids, scores = self.model.recommend(user_idx, self.R[user_idx], N=N)
        return list(zip(ids.tolist(), scores.tolist()))

    def explain(self, user_idx, recommended_item_idxs, train, movies, top_n=20):
        itf            = self.model.item_factors
        idx_to_movieid = train.drop_duplicates("item_idx").set_index("item_idx")["movieId"].to_dict()
        item_to_title  = movies.drop_duplicates("movieId").set_index("movieId")["title"].to_dict()

        user_ratings   = (train[train["user_idx"] == user_idx]
                          .sort_values("rating", ascending=False))
        rated_idxs     = user_ratings["item_idx"].values.astype(int)
        actual_ratings = {int(r["item_idx"]): r["rating"]
                          for _, r in user_ratings.iterrows()}

        if len(rated_idxs) == 0:
            return []

        rec_emb   = itf[recommended_item_idxs]
        rated_emb = itf[rated_idxs]

        rec_norm   = rec_emb   / (np.linalg.norm(rec_emb,   axis=1, keepdims=True) + 1e-8)
        rated_norm = rated_emb / (np.linalg.norm(rated_emb, axis=1, keepdims=True) + 1e-8)
        agg = (rec_norm @ rated_norm.T).sum(axis=0)  # aggregate similarity across all recs

        top_j  = np.argsort(agg)[::-1][:top_n]
        result = []
        for j in top_j:
            movie_id = idx_to_movieid.get(int(rated_idxs[j]))
            title    = item_to_title.get(movie_id, f"MovieID {movie_id}")
            result.append((title, actual_ratings.get(int(rated_idxs[j]), 0.0)))
        return result