"""
Ranking evaluation: Precision@K, Recall@K, NDCG@K on the test set.
Run from the repo root: python EVALUATIONS/eval_ranking.py
Requires CODE/weights/ to be populated by running CODE/main.py first.
"""
import sys, os, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "CODE"))
os.environ["OPENBLAS_NUM_THREADS"] = "1"

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "DATA")
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "CODE", "weights")

from src.data import load_1m, time_split
from src.evaluate import evaluate_ranking

def load(name):
    path = os.path.join(WEIGHTS_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Weight file not found: {path}\nRun CODE/main.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)

print("Loading ml-1m...")
ratings, movies, users = load_1m(path=os.path.join(DATA_DIR, "ml-1m"))

print("Splitting...")
train, val, test = time_split(ratings)

print("\n--- Ranking Metrics (Test Set, threshold=4.0) ---")
print("Evaluating ALS...")
print("ALS:   ", evaluate_ranking(load("als"), train, test, movies, k_list=[5, 10, 20], max_users=500))
print("Evaluating KNN...")
print("KNN:   ", evaluate_ranking(load("knn"), train, test, movies, k_list=[5, 10, 20], max_users=500))
print("Evaluating NeuMF...")
print("NeuMF: ", evaluate_ranking(load("neumf"), train, test, movies, k_list=[5, 10, 20], max_users=500))
