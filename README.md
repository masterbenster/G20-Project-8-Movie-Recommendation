# G20 Project 8 — Movie Recommendation System

A comparative study of collaborative filtering algorithms for movie recommendation, evaluated on the MovieLens 1M and 10M datasets. The system implements five models ranging from simple baselines to a deep neural network, with explainability built into each model's inference output.

**Models implemented:**
- Global Mean & Bias Model (baselines)
- ALS — Alternating Least Squares matrix factorization
- ItemKNN — Item-based collaborative filtering
- NeuMF — Neural Collaborative Filtering (GMF + MLP hybrid)

**Key results (MovieLens 10M, NDCG@10):**

| Model | NDCG@10 |
|---|---|
| Global Mean | 0.053 |
| Bias Model | 0.217 |
| ALS | 0.264 |
| ItemKNN | 0.254 |
| NeuMF | **0.597** |

---

## Project Structure

```
CODE/
  src/
    cli.py          # Inference CLI
    als.py          # ALS model
    knn.py          # ItemKNN model
    neumf.py        # NeuMF model
    baselines.py    # GlobalMean and BiasModel
    data.py         # Data loading and splitting
    evaluate.py     # RMSE, MAE, Precision/Recall/NDCG
  train.py          # Train all models on MovieLens 1M
  train_10m.py      # Train all models on MovieLens 10M
  weights/          # Saved 1M model weights
  weights_10m/      # Saved 10M model weights
DATA/
  ml-1m/            # MovieLens 1M dataset
  ml-10M100K/       # MovieLens 10M dataset
EVALUATIONS/        # Standalone evaluation scripts
demo_1m.sh          # Demo script for MovieLens 1M
demo_10m.sh         # Demo script for MovieLens 10M
```

---

## Setup

### Prerequisites

- Python 3.12

### 1. Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

**NVIDIA CUDA (recommended for most users):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.4 pandas scikit-learn scipy joblib pyarrow tqdm implicit
```

**AMD ROCm (Linux, Windows WSL2):**

The repo includes pre-built ROCm 6.4.2 wheels for Python 3.12. Install them directly:

```bash
pip install \
  torch-2.6.0+rocm6.4.2.git76481f7c-cp312-cp312-linux_x86_64.whl \
  torchvision-0.21.0+rocm6.4.2.git4040d51f-cp312-cp312-linux_x86_64.whl \
  torchaudio-2.6.0+rocm6.4.2.gitd8831425-cp312-cp312-linux_x86_64.whl \
  pytorch_triton_rocm-3.2.0+rocm6.4.2.git7e948ebf-cp312-cp312-linux_x86_64.whl
pip install -r CODE/requirements_win11-amd.txt
```

**CPU only:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy==1.26.4 pandas scikit-learn scipy joblib pyarrow tqdm implicit
```

### 3. Download datasets

The datasets are already included in `DATA/`. If you need to re-download them:

- [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) → extract into `DATA/ml-1m/`
- [MovieLens 10M](https://grouplens.org/datasets/movielens/10m/) → extract into `DATA/ml-10M100K/`

---

## Training

Train all five models on MovieLens 1M (~6 minutes):

```bash
python CODE/train.py
```

Train all five models on MovieLens 10M (~1 hour w/Radeon RX 9070 XT):

```bash
python CODE/train_10m.py
```

Weights are saved to `CODE/weights/` (1M) and `CODE/weights_10m/` (10M). Train/val/test splits are cached as Parquet files in the dataset directory so subsequent runs skip re-splitting.

---

## Inference

```bash
# Single recommendation query (run from repo root)
.venv/bin/python CODE/src/cli.py --user 42 --model neumf --topn 5
.venv/bin/python CODE/src/cli.py --user 42 --model als   --topn 5 --dataset 10m
.venv/bin/python CODE/src/cli.py --user 42 --model knn   --topn 10
```

**Arguments:**

| Argument | Options | Default | Description |
|---|---|---|---|
| `--user` | integer | required | User ID (1–6040 for 1M, 1–71567 for 10M) |
| `--model` | `als`, `knn`, `neumf` | `als` | Model to use |
| `--topn` | integer | 10 | Number of recommendations |
| `--dataset` | `1m`, `10m` | `1m` | Dataset to load weights for |

Each model outputs the top-N recommended movies along with the top-rated items from that user's history that drove the recommendations.

### Demo scripts

```bash
./demo_1m.sh   # Runs NeuMF for User 42 on MovieLens 1M
./demo_10m.sh  # Runs NeuMF for User 42 on MovieLens 10M
```

---

## Slopflix Web UI

A Netflix-style web interface for browsing recommendations, built with FastAPI and HTMX.

### Setup

Create a `.env` file in the repo root with your [TMDB API key](https://www.themoviedb.org/settings/api) (free):

```
TMDB_API_KEY=your_key_here
```

Install the additional dependencies:

```bash
.venv/bin/pip install -r CODE/slopflix/requirements.txt
```

### Running

```bash
.venv/bin/uvicorn CODE.slopflix.main:app
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

The server preloads the MovieLens 1M dataset and all three models (NeuMF, ALS, ItemKNN) on startup — this takes ~30 seconds but keeps every subsequent request fast. Movie posters are fetched from TMDB and cached in memory for the lifetime of the server process.

---

## Evaluation

Standalone evaluation scripts are in `EVALUATIONS/`:

```bash
# Rating prediction (RMSE, MAE)
python EVALUATIONS/eval_rating_1m.py
python EVALUATIONS/eval_rating_10m.py

# Ranking metrics (Precision, Recall, NDCG @5/10/20)
python EVALUATIONS/eval_ranking_1m.py
python EVALUATIONS/eval_ranking_10m.py
```

Ranking evaluation uses sampled evaluation: each test item is ranked against 99 randomly sampled unseen items, evaluated over up to 500 users.
