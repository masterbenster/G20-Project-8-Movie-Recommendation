import os, sys, re, asyncio
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["OPENBLAS_NUM_THREADS"]  = "1"

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import pandas as pd
import httpx

from src.data import load_1m, load_10m, load_splits, save_splits, time_split

# ── Paths ──────────────────────────────────────────────────────────────────────
_SLOPFLIX_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR     = os.path.dirname(_SLOPFLIX_DIR)
_ROOT_DIR     = os.path.dirname(_CODE_DIR)
_DATA_DIR     = os.path.join(_ROOT_DIR, "DATA")

DATASET_DIRS = {
    "1m":  os.path.join(_DATA_DIR, "ml-1m"),
    "10m": os.path.join(_DATA_DIR, "ml-10M100K"),
}
WEIGHTS_DIRS = {
    "1m":  os.path.join(_CODE_DIR, "weights"),
    "10m": os.path.join(_CODE_DIR, "weights_10m"),
}

# ── Config ─────────────────────────────────────────────────────────────────────
TMDB_API_KEY    = os.environ.get("TMDB_API_KEY", "")
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"

# ── In-memory caches ───────────────────────────────────────────────────────────
_data_cache   = {}  # dataset -> dict
_model_cache  = {}  # (dataset, model) -> model object
_poster_cache = {}  # title -> poster URL string

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    print("Loading 1M dataset and models...")
    _load_data("1m")
    for model in ("neumf", "als", "knn"):
        _load_model("1m", model)
    print("Ready.")
    yield

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory=os.path.join(_SLOPFLIX_DIR, "templates"))


# ── Data & model helpers ───────────────────────────────────────────────────────

def _load_data(dataset: str) -> dict:
    if dataset in _data_cache:
        return _data_cache[dataset]

    ds_dir = DATASET_DIRS[dataset]
    if dataset == "1m":
        ratings, movies, _ = load_1m(path=ds_dir)
    else:
        ratings, movies, _ = load_10m(path=ds_dir)

    cached = load_splits(ds_dir)
    if cached:
        train, _val, _test = cached
    else:
        train, _val, _test = time_split(ratings)
        save_splits(train, _val, _test, ds_dir)

    user_map = train.drop_duplicates("userId").set_index("userId")["user_idx"].to_dict()

    _data_cache[dataset] = {
        "train":    train,
        "movies":   movies,
        "ratings":  ratings,
        "user_map": user_map,
    }
    return _data_cache[dataset]


def _load_model(dataset: str, model_name: str):
    key = (dataset, model_name)
    if key not in _model_cache:
        path = os.path.join(WEIGHTS_DIRS[dataset], f"{model_name}.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No weights found at {os.path.relpath(path)}. Run train.py first."
            )
        _model_cache[key] = joblib.load(path)
    return _model_cache[key]


def _get_recs(model, user_idx, train, movies, ratings, n=10):
    seen       = set(ratings[ratings["user_idx"] == user_idx]["item_idx"].values)
    all_items  = np.arange(train["item_idx"].max() + 1)
    candidates = np.array([i for i in all_items if i not in seen])

    df_cand    = pd.DataFrame({"user_idx": user_idx, "item_idx": candidates})
    scores     = model.predict(df_cand)
    order      = np.argsort(scores)[::-1][:n]
    top_idx    = candidates[order]
    top_scores = scores[order]

    id_map  = train.drop_duplicates("item_idx").set_index("item_idx")["movieId"].to_dict()
    results = []
    for idx, score in zip(top_idx, top_scores):
        movie_id = id_map.get(idx)
        if movie_id is None:
            continue
        row   = movies[movies["movieId"] == movie_id]["title"].values
        title = row[0] if len(row) else f"MovieID {movie_id}"
        results.append((title, round(float(score), 3)))
    return results


def _get_history(model, user_idx, recs, train, movies):
    if not hasattr(model, "explain"):
        return []
    title_to_idx = (
        train.drop_duplicates("item_idx")
             .merge(movies, on="movieId")[["title", "item_idx"]]
             .set_index("title")["item_idx"].to_dict()
    )
    rec_idxs = [title_to_idx[t] for t, _ in recs if t in title_to_idx]
    if len(rec_idxs) != len(recs):
        return []
    return model.explain(user_idx, rec_idxs, train, movies)


# ── TMDB poster fetching ───────────────────────────────────────────────────────

_YEAR_RE = re.compile(r"^(.*?)\s*\((\d{4})\)\s*$")


async def _fetch_poster(title: str, client: httpx.AsyncClient) -> str:
    if title in _poster_cache:
        return _poster_cache[title]
    if not TMDB_API_KEY:
        _poster_cache[title] = ""
        return ""

    m = _YEAR_RE.match(title)
    clean, year = (m.group(1).strip(), m.group(2)) if m else (title, None)

    params: dict = {"api_key": TMDB_API_KEY, "query": clean, "include_adult": "false"}
    if year:
        params["primary_release_year"] = year

    try:
        r       = await client.get(TMDB_SEARCH_URL, params=params, timeout=5.0)
        results = r.json().get("results", [])
        poster  = (TMDB_IMAGE_BASE + results[0]["poster_path"]) if results and results[0].get("poster_path") else ""
    except Exception:
        poster = ""

    _poster_cache[title] = poster
    return poster


async def _fetch_all_posters(titles: list) -> list:
    async with httpx.AsyncClient() as client:
        return await asyncio.gather(*[_fetch_poster(t, client) for t in titles])


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/recommend", response_class=HTMLResponse)
async def recommend(
    request:  Request,
    user_id:  int = Form(...),
    model:    str = Form("neumf"),
    dataset:  str = Form("1m"),
    topn:     int = Form(10),
):
    try:
        data     = _load_data(dataset)
        mdl      = _load_model(dataset, model)
        user_map = data["user_map"]

        if user_id not in user_map:
            lo, hi = min(user_map), max(user_map)
            return HTMLResponse(
                f'<p class="error-msg">User {user_id} not found. Valid range: {lo}–{hi}.</p>'
            )

        user_idx = user_map[user_id]
        recs     = _get_recs(mdl, user_idx, data["train"], data["movies"], data["ratings"], n=topn)
        history  = _get_history(mdl, user_idx, recs, data["train"], data["movies"])
        posters  = await _fetch_all_posters([t for t, _ in recs])

        rec_data = [(title, score, poster) for (title, score), poster in zip(recs, posters)]

        return templates.TemplateResponse(request, "results.html", {
            "user_id": user_id,
            "model":   model,
            "dataset": dataset,
            "topn":    topn,
            "recs":    rec_data,
            "history": history,
        })

    except FileNotFoundError as e:
        return HTMLResponse(f'<p class="error-msg">{e}</p>')
    except Exception as e:
        return HTMLResponse(f'<p class="error-msg">Error: {e}</p>')
