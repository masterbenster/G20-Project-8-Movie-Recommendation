import os
import joblib
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from src.data import load_1m, time_split
from src.baselines import GlobalMean, BiasModel
from src.als import ALSModel
from src.knn import ItemKNN
from src.neumf import NeuMFModel

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "DATA")
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

def save(model, name):
    path = os.path.join(WEIGHTS_DIR, f"{name}.joblib")
    joblib.dump(model, path, compress=("bz2", 3))
    print(f"  Saved -> weights/{name}.joblib")

print("Loading ml-1m...")
ratings, movies, users = load_1m(path=os.path.join(DATA_DIR, "ml-1m"))

print("Splitting...")
train, val, test = time_split(ratings)

print("\nTraining GlobalMean...")
gm = GlobalMean()
gm.fit(train)
save(gm, "global_mean")

print("Training BiasModel...")
bm = BiasModel()
bm.fit(train)
save(bm, "bias_model")

print("Training ALS...")
als = ALSModel(factors=50, iterations=20, regularization=0.1)
als.fit(train)
save(als, "als")

print("Training ItemKNN...")
knn = ItemKNN(k=20)
knn.fit(train)
save(knn, "knn")

print("Training NeuMF...")
neumf = NeuMFModel(emb_dim=32, layers=[64, 32, 16], epochs=10, lr=0.001, batch_size=1024)
neumf.fit(train)
save(neumf, "neumf")

print("\nAll models trained and saved to CODE/weights/.")
print("Run EVALUATIONS/eval_rating.py or EVALUATIONS/eval_ranking.py for results.")