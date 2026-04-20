import os
import joblib
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from threadpoolctl import threadpool_limits
threadpool_limits(1, "blas")

from src.data import load_1m, load_splits, save_splits, time_split
from src.baselines import GlobalMean, BiasModel
from src.als import ALSModel
from src.knn import ItemKNN
from src.neumf import NeuMFModel
from src.evaluate import evaluate_rating, evaluate_ranking

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "DATA")
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
DATASET_DIR = os.path.join(DATA_DIR, "ml-1m")
os.makedirs(WEIGHTS_DIR, exist_ok=True)


def save(model, name):
    path = os.path.join(WEIGHTS_DIR, f"{name}.joblib")
    joblib.dump(model, path, compress=("bz2", 3))
    print(f"  Saved -> weights/{name}.joblib\n")


def train_eval(name, model, train, val, movies):
    print(f"Training {name}...")
    model.fit(train)
    if getattr(model, "rating_model", True):
        val_metrics = evaluate_rating(model, val)
        print(f"  Val  -> RMSE: {val_metrics['RMSE']}  MAE: {val_metrics['MAE']}")
    else:
        print(f"  Val  -> (BCE ranking model — no RMSE/MAE)")
    save(model, name.lower().replace(" ", "_"))
    return model


# ── Load data ────────────────────────────────────────────────────────────────
print("Loading ml-1m...")
ratings, movies, users = load_1m(path=DATASET_DIR)

cached = load_splits(DATASET_DIR)
if cached:
    print("  Found cached splits, loading...\n")
    train, val, test = cached
else:
    print("  Splitting...")
    train, val, test = time_split(ratings)
    save_splits(train, val, test, DATASET_DIR)

# ── Train all models (val metrics shown after each for hyperparameter tuning) ─
models = {}
models["global_mean"] = train_eval("GlobalMean", GlobalMean(),  train, val, movies)
models["bias_model"]  = train_eval("BiasModel",  BiasModel(),   train, val, movies)
models["als"]         = train_eval("ALS",         ALSModel(factors=128, iterations=30, regularization=0.05), train, val, movies)
models["knn"]         = train_eval("KNN",         ItemKNN(k=20), train, val, movies)
models["neumf"]       = train_eval("NeuMF",       NeuMFModel(emb_dim=64, layers=[128, 64, 32]), train, val, movies)

# ── Final test evaluation (run once, after hyperparams are locked in) ─────────
print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print("\n-- Rating Prediction (RMSE / MAE) --")
for name, model in models.items():
    if not getattr(model, "rating_model", True):
        print(f"  {name:<12} (BCE ranking model — skipped)")
        continue
    m = evaluate_rating(model, test)
    print(f"  {name:<12} RMSE: {m['RMSE']}  MAE: {m['MAE']}")

print("\n-- Ranking (Precision / Recall / NDCG @K) --")
for name, model in models.items():
    m = evaluate_ranking(model, train, test, movies, k_list=[5, 10, 20], max_users=500)
    print(f"  {name}")
    for k in [5, 10, 20]:
        print(f"    @{k:<3}  P: {m[f'P@{k}']}  R: {m[f'R@{k}']}  NDCG: {m[f'NDCG@{k}']}")
