"""
Rating prediction evaluation: RMSE and MAE on the test set (MovieLens 10M).
Run from the repo root: python EVALUATIONS/eval_rating_10m.py
Requires CODE/weights_10m/ to be populated by running CODE/main.py first.
"""
import sys, os, joblib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "CODE"))
os.environ["OPENBLAS_NUM_THREADS"] = "1"

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "DATA")
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "CODE", "weights_10m")

from src.data import load_splits, load_10m, time_split
from src.evaluate import evaluate_rating

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

def eval_rating(name):
    model = load(name)
    if getattr(model, "rating_model", True) is False:
        return "(BCE ranking model — skipped)"
    return evaluate_rating(model, test)

print("\n--- Rating Prediction (Test Set, 10M) ---")
print("GlobalMean:", eval_rating("globalmean"))
print("BiasModel: ", eval_rating("biasmodel"))
print("ALS:       ", eval_rating("als"))
print("ItemKNN:   ", eval_rating("knn"))
print("NeuMF:     ", eval_rating("neumf"))
