"""
Ranking evaluation: Precision@K, Recall@K, NDCG@K on the test set (MovieLens 10M).
Run from the repo root: python EVALUATIONS/eval_ranking_10m.py
Requires CODE/weights_10m/ to be populated by running CODE/main.py first.
"""
import sys, os, joblib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "CODE"))
os.environ["OPENBLAS_NUM_THREADS"] = "1"

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "DATA")
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "CODE", "weights_10m")

from src.data import load_splits, load_10m, time_split
from src.evaluate import evaluate_ranking

def load(name):
    path = os.path.join(WEIGHTS_DIR, f"{name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Weight file not found: {path}\nRun CODE/main.py first.")
    return joblib.load(path)

dataset_dir = os.path.join(DATA_DIR, "ml-10M100K")
splits = load_splits(dataset_dir)
if splits:
    train, val, test = splits
else:
    print("Loading ml-10M100K...")
    ratings, _, _ = load_10m(path=dataset_dir)
    print("Splitting...")
    train, val, test = time_split(ratings)

KWARGS = dict(k_list=[5, 10, 20], max_users=500)

print("\n--- Ranking Metrics (Test Set, threshold=4.0, 10M) ---")
print("Evaluating GlobalMean...")
print("GlobalMean:", evaluate_ranking(load("globalmean"), train, test, None, **KWARGS))
print("Evaluating BiasModel...")
print("BiasModel: ", evaluate_ranking(load("biasmodel"), train, test, None, **KWARGS))
print("Evaluating ALS...")
print("ALS:       ", evaluate_ranking(load("als"),       train, test, None, **KWARGS))
print("Evaluating KNN...")
print("KNN:       ", evaluate_ranking(load("knn"),       train, test, None, **KWARGS))
print("Evaluating NeuMF...")
print("NeuMF:     ", evaluate_ranking(load("neumf"),     train, test, None, **KWARGS))
