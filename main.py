import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from src.data import load_1m, time_split
from src.baselines import GlobalMean, BiasModel
from src.als import ALSModel
from src.knn import ItemKNN
from src.evaluate import evaluate_rating, evaluate_ranking

print("Loading ml-1m...")
ratings, movies, users = load_1m()

print("Splitting...")
train, val, test = time_split(ratings)

print("\n--- Rating Prediction (Val Set) ---")
gm = GlobalMean()
gm.fit(train)
print("GlobalMean:", evaluate_rating(gm, val))

bm = BiasModel()
bm.fit(train)
print("BiasModel: ", evaluate_rating(bm, val))

als = ALSModel(factors=50, iterations=20, regularization=0.1)
als.fit(train)
print("ALS:       ", evaluate_rating(als, val))

knn = ItemKNN(k=20)
knn.fit(train)
print("ItemKNN:   ", evaluate_rating(knn, val))

print("\n--- Ranking Metrics (Test Set, threshold=4.0) ---")
print("Evaluating ALS...")
print("ALS:   ", evaluate_ranking(als, train, test, movies, k_list=[5, 10, 20]))
print("Evaluating KNN...")
print("KNN:   ", evaluate_ranking(knn, train, test, movies, k_list=[5, 10, 20]))