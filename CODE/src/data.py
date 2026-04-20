import pandas as pd

def load_1m(path="ml-1m"):
    ratings = pd.read_csv(f"{path}/ratings.dat", sep="::", engine="python",
                          names=["userId","movieId","rating","timestamp"])
    movies  = pd.read_csv(f"{path}/movies.dat", sep="::", engine="python",
                          encoding="latin-1",
                          names=["movieId","title","genres"])
    users   = pd.read_csv(f"{path}/users.dat", sep="::", engine="python",
                          names=["userId","gender","age","occupation","zip"])

    ratings["user_idx"] = ratings["userId"].astype("category").cat.codes
    ratings["item_idx"] = ratings["movieId"].astype("category").cat.codes
    return ratings, movies, users

def load_10m(path="ml-10M100K"):
    ratings = pd.read_csv(f"{path}/ratings.dat", sep="::", engine="python",
                          names=["userId","movieId","rating","timestamp"])
    movies  = pd.read_csv(f"{path}/movies.dat", sep="::", engine="python",
                          encoding="utf-8",
                          names=["movieId","title","genres"])
    tags    = pd.read_csv(f"{path}/tags.dat", sep="::", engine="python",
                          encoding="utf-8",
                          names=["userId","movieId","tag","timestamp"])

    ratings["user_idx"] = ratings["userId"].astype("category").cat.codes
    ratings["item_idx"] = ratings["movieId"].astype("category").cat.codes
    return ratings, movies, tags

def time_split(df):
    df = df.sort_values(["userId","timestamp"])
    df["rank"] = df.groupby("userId").cumcount(ascending=False)
    train = df[df["rank"] >= 2].copy()
    val   = df[df["rank"] == 1].copy()
    test  = df[df["rank"] == 0].copy()
    return train, val, test