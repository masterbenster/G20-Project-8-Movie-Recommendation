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