import numpy as np

class GlobalMean:
    def fit(self, train):
        self.mu = train["rating"].mean()

    def predict(self, df):
        return np.full(len(df), self.mu)


class BiasModel:
    def fit(self, train):
        self.mu = train["rating"].mean()
        self.user_bias = train.groupby("user_idx")["rating"].mean() - self.mu
        self.item_bias = train.groupby("item_idx")["rating"].mean() - self.mu

    def predict(self, df):
        u = df["user_idx"].map(self.user_bias).fillna(0)
        i = df["item_idx"].map(self.item_bias).fillna(0)
        return (self.mu + u + i).clip(0.5, 5.0).values