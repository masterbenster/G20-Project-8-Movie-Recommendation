import argparse
import json
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from scripts.ranking_eval import build_candidate_groups, default_max_ranking_events, read_csv_gz
except ModuleNotFoundError:
    from ranking_eval import build_candidate_groups, default_max_ranking_events, read_csv_gz


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results" / "neumf"


def _group_candidate_events(test_negs_df: pd.DataFrame):
    """
    test_negs_df schema:
      user_index, pos_movie_index, neg_movie_index

    Returns:
      list[(user_index, pos_movie_index, neg_movie_indices_array)]
    """
    df = test_negs_df.sort_values(["event_id", "neg_movie_index"], kind="stable")
    grouped = df.groupby("event_id", sort=False)
    out = []
    for event_id, g in grouped:
        u = int(g["user_index"].iloc[0])
        pos_movie = int(g["pos_movie_index"].iloc[0])
        negs = g["neg_movie_index"].to_numpy(dtype=np.int32, copy=False)
        out.append((u, pos_movie, negs))
    return out


def evaluate_topn_ndcg_onepos(
    model: nn.Module,
    *,
    device: torch.device,
    test_negs_df: pd.DataFrame,
    ks: list[int],
    num_movies: int,
    batch_users: int = 512,
) -> dict[str, float]:
    """
    For each user u:
      candidate set = [pos] + negs (from test_negs_df)
      compute logits for all candidates
      compute Precision@K, Recall@K, NDCG@K (one-positive-per-event)
    """
    model.eval()
    groups = _group_candidate_events(test_negs_df)

    prec_sum = {k: 0.0 for k in ks}
    rec_sum = {k: 0.0 for k in ks}
    ndcg_sum = {k: 0.0 for k in ks}

    # Evaluate in batches by held-out events.
    for start in range(0, len(groups), batch_users):
        batch_groups = groups[start : start + batch_users]
        user_rep = []
        item_rep = []
        pos_items = []
        group_sizes = []
        for u, pos_movie, negs in batch_groups:
            # Candidate set includes the positive once + negatives.
            # Remove any accidental duplicates of pos among negatives.
            if negs.size > 0:
                negs = negs[negs != pos_movie]
            candidates = np.concatenate(([pos_movie], negs.astype(np.int32, copy=False)))
            pos_items.append(pos_movie)
            group_sizes.append(len(candidates))
            user_rep.append(np.full(len(candidates), u, dtype=np.int32))
            item_rep.append(candidates)

        user_rep = np.concatenate(user_rep, axis=0)
        item_rep = np.concatenate(item_rep, axis=0)

        # Forward pass for all candidates at once.
        with torch.no_grad():
            u_t = torch.from_numpy(user_rep).to(device=device, dtype=torch.long)
            i_t = torch.from_numpy(item_rep).to(device=device, dtype=torch.long)
            logits = model(u_t, i_t).detach().cpu().numpy().astype(np.float32, copy=False)

        # Compute rank_pos for each user separately.
        offset = 0
        for idx_u, (_u, _pos_movie, _negs) in enumerate(batch_groups):
            pos_movie = int(pos_items[idx_u])
            size = int(group_sizes[idx_u])
            cand_items = item_rep[offset : offset + size].astype(np.int32, copy=False)
            cand_scores = logits[offset : offset + size]
            offset += size

            # Sort: higher score first, tie-break by higher movie_index first (deterministic).
            order = np.lexsort((-cand_items, -cand_scores))
            pos_in_order = int(np.where(cand_items[order] == pos_movie)[0][0])
            rank_pos = pos_in_order + 1  # 1-indexed

            for k in ks:
                if rank_pos <= k:
                    prec_sum[k] += 1.0 / float(k)
                    rec_sum[k] += 1.0
                    ndcg_sum[k] += 1.0 / float(np.log2(rank_pos + 1.0))

    num_users = float(len(groups))
    return {
        **{f"precision@{k}": prec_sum[k] / num_users for k in ks},
        **{f"recall@{k}": rec_sum[k] / num_users for k in ks},
        **{f"ndcg@{k}": ndcg_sum[k] / num_users for k in ks},
    }


class NeuMF(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        *,
        embed_dim_gmf: int = 32,
        embed_dim_mlp: int = 32,
        mlp_layers: list[int] = (64, 32, 16),
        dropout: float = 0.2,
    ):
        super().__init__()

        self.user_gmf = nn.Embedding(num_users, embed_dim_gmf)
        self.item_gmf = nn.Embedding(num_items, embed_dim_gmf)

        self.user_mlp = nn.Embedding(num_users, embed_dim_mlp)
        self.item_mlp = nn.Embedding(num_items, embed_dim_mlp)

        mlp_in = embed_dim_mlp * 2
        mlp = []
        prev = mlp_in
        for h in mlp_layers:
            mlp.append(nn.Linear(prev, h))
            mlp.append(nn.ReLU())
            mlp.append(nn.Dropout(dropout))
            prev = h
        self.mlp = nn.Sequential(*mlp)

        # GMF produces embed_dim_gmf features; MLP produces last dim in mlp_layers.
        self.out = nn.Linear(embed_dim_gmf + mlp_layers[-1], 1)

        # Initialize embeddings with small normal noise.
        nn.init.normal_(self.user_gmf.weight, std=0.01)
        nn.init.normal_(self.item_gmf.weight, std=0.01)
        nn.init.normal_(self.user_mlp.weight, std=0.01)
        nn.init.normal_(self.item_mlp.weight, std=0.01)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        # user_idx/item_idx: shape (batch,)
        gmf_u = self.user_gmf(user_idx)
        gmf_i = self.item_gmf(item_idx)
        gmf_out = gmf_u * gmf_i

        mlp_u = self.user_mlp(user_idx)
        mlp_i = self.item_mlp(item_idx)
        mlp_in = torch.cat([mlp_u, mlp_i], dim=1)
        mlp_out = self.mlp(mlp_in)

        z = torch.cat([gmf_out, mlp_out], dim=1)
        logits = self.out(z).squeeze(1)
        return logits


def build_user_rated_mask(ratings_df: pd.DataFrame, num_users: int, num_items: int) -> np.ndarray:
    """
    Returns bool mask of shape (num_users, num_items) where True means "user rated item".
    """
    mask = np.zeros((num_users, num_items), dtype=bool)
    u = ratings_df["user_index"].to_numpy(dtype=np.int32, copy=False)
    i = ratings_df["movie_index"].to_numpy(dtype=np.int32, copy=False)
    mask[u, i] = True
    return mask


def filter_positive_interactions(ratings_df: pd.DataFrame, *, positive_threshold: float) -> pd.DataFrame:
    """
    Keep only interactions that count as positive implicit feedback for NeuMF training.
    """
    return ratings_df[ratings_df["rating"] >= float(positive_threshold)].copy()


def sample_negatives_for_batch(
    rng: np.random.Generator,
    *,
    user_idx: np.ndarray,
    pos_item_idx: np.ndarray,
    rated_mask: np.ndarray,
    num_items: int,
) -> np.ndarray:
    """
    For each (u, pos_item), sample one negative item not rated by u.
    Vectorized rejection sampling with minimal loops.
    """
    batch = user_idx.shape[0]
    neg = rng.integers(0, num_items, size=batch, dtype=np.int32)

    # Ensure neg is not a rated item and not equal to pos_item.
    rated = rated_mask[user_idx, neg]
    same = neg == pos_item_idx
    bad = rated | same

    # Rejection sample until all are valid.
    # Expected to converge quickly for ML-1M.
    while bool(np.any(bad)):
        idx = np.where(bad)[0]
        neg[idx] = rng.integers(0, num_items, size=len(idx), dtype=np.int32)
        rated = rated_mask[user_idx, neg]
        same = neg == pos_item_idx
        bad = rated | same
    return neg


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuMF training + top-N evaluation.")
    parser.add_argument("--dataset", choices=["1m", "10m"], default="1m")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--neg-ratio", type=int, default=1, help="Negatives per positive")
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--mlp", type=str, default="64,32,16")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=4.0,
        help="Minimum explicit rating treated as a positive interaction for NeuMF training.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ks", type=str, default="5,10,20")
    parser.add_argument(
        "--max-ranking-events",
        type=int,
        default=None,
        help="Maximum held-out events to evaluate for ranking. Default: 20000 for both 1m and 10m.",
    )
    args = parser.parse_args()

    if args.dataset != "1m":
        raise SystemExit("NeuMF script currently supports MovieLens 1M only (mask-based negative sampling).")

    ks = [int(x) for x in args.ks.split(",") if x]
    mlp_layers = [int(x) for x in args.mlp.split(",") if x]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processed_dir = PROCESSED_DIR / args.dataset
    meta = json.load(open(processed_dir / "meta.json", "r"))
    num_users = int(meta["num_users"])
    num_items = int(meta["num_movies"])

    train_df = read_csv_gz(
        processed_dir / "train.csv.gz",
        dtypes={"user_index": np.int32, "movie_index": np.int32, "rating": np.float32, "timestamp": np.int64},
    )
    val_df = read_csv_gz(
        processed_dir / "val.csv.gz",
        dtypes={"user_index": np.int32, "movie_index": np.int32, "rating": np.float32, "timestamp": np.int64},
    )
    test_df = read_csv_gz(
        processed_dir / "test.csv.gz",
        dtypes={"user_index": np.int32, "movie_index": np.int32, "rating": np.float32, "timestamp": np.int64},
    )
    ratings_clean_df = read_csv_gz(
        processed_dir / "ratings_clean.csv.gz",
        dtypes={"user_index": np.int32, "movie_index": np.int32, "rating": np.float32, "timestamp": np.int64},
    )
    max_ranking_events = default_max_ranking_events(args.dataset) if args.max_ranking_events is None else args.max_ranking_events
    _val_events_df, val_negs_df, val_eval_summary = build_candidate_groups(
        split_df=val_df,
        rated_source_df=ratings_clean_df,
        num_movies=num_items,
        num_negatives=99,
        seed=args.seed + 500,
        max_events=max_ranking_events,
    )
    _test_events_df, test_negs_df, test_eval_summary = build_candidate_groups(
        split_df=test_df,
        rated_source_df=ratings_clean_df,
        num_movies=num_items,
        num_negatives=99,
        seed=args.seed + 600,
        max_events=max_ranking_events,
    )
    print(f"[prep] dataset={args.dataset} users={num_users} items={num_items} train_rows={len(train_df)}")

    train_pos_df = filter_positive_interactions(train_df, positive_threshold=args.positive_threshold)
    if train_pos_df.empty:
        raise SystemExit(
            f"NeuMF positive-threshold={args.positive_threshold} produced 0 training positives for dataset={args.dataset}."
        )
    print(
        f"[prep] dataset={args.dataset} positive_threshold={args.positive_threshold} "
        f"train_positive_rows={len(train_pos_df)}"
    )

    # Training negatives avoid any item seen in TRAIN, including low-rated items.
    # Validation/test candidate negatives still exclude the user's full known history
    # through the shared ranking evaluation pipeline, so this is a deliberate
    # train-vs-eval compromise that avoids leakage from held-out interactions.
    rated_mask = build_user_rated_mask(train_df, num_users=num_users, num_items=num_items)

    # Build positive training tensors from strong-feedback interactions only.
    u_pos = train_pos_df["user_index"].to_numpy(dtype=np.int32, copy=False)
    i_pos = train_pos_df["movie_index"].to_numpy(dtype=np.int32, copy=False)
    n_pos = int(u_pos.shape[0])

    # Model.
    model = NeuMF(
        num_users=num_users,
        num_items=num_items,
        embed_dim_gmf=args.embed_dim,
        embed_dim_mlp=args.embed_dim,
        mlp_layers=mlp_layers,
        dropout=args.dropout,
    ).to(device=device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop.
    best = {"ndcg@10": -1.0, "epoch": -1, "val_metrics": None}
    best_state_dict = None
    steps_per_epoch = max(1, n_pos // args.batch_size)

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Sample random positive indices each step.
        for _ in range(steps_per_epoch):
            idx = rng.integers(0, n_pos, size=args.batch_size, dtype=np.int32)
            u_batch = u_pos[idx]
            i_batch_pos = i_pos[idx]

            # Negatives.
            i_batch_negs = []
            for _neg in range(args.neg_ratio):
                i_batch_negs.append(
                    sample_negatives_for_batch(
                        rng,
                        user_idx=u_batch,
                        pos_item_idx=i_batch_pos,
                        rated_mask=rated_mask,
                        num_items=num_items,
                    )
                )
            i_batch_negs = np.stack(i_batch_negs, axis=1)  # (batch, neg_ratio)

            # Create combined batch: positives + negatives.
            # Positives.
            u_t = torch.from_numpy(u_batch).to(device=device, dtype=torch.long)
            i_t_pos = torch.from_numpy(i_batch_pos).to(device=device, dtype=torch.long)
            y_pos = torch.ones(args.batch_size, device=device, dtype=torch.float32)
            logits_pos = model(u_t, i_t_pos)

            # Negatives (flatten across neg_ratio).
            u_t_neg = torch.from_numpy(np.repeat(u_batch, args.neg_ratio)).to(device=device, dtype=torch.long)
            i_t_neg = torch.from_numpy(i_batch_negs.reshape(-1)).to(device=device, dtype=torch.long)
            y_neg = torch.zeros(args.batch_size * args.neg_ratio, device=device, dtype=torch.float32)
            logits_neg = model(u_t_neg, i_t_neg)

            logits = torch.cat([logits_pos, logits_neg], dim=0)
            y = torch.cat([y_pos, y_neg], dim=0)

            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # Validation after each epoch.
        val_metrics = evaluate_topn_ndcg_onepos(
            model,
            device=device,
            test_negs_df=val_negs_df,
            ks=ks,
            num_movies=num_items,
        )
        ndcg10 = float(val_metrics.get("ndcg@10", 0.0))
        print(f"[epoch {epoch}] val ndcg@10={ndcg10:.6f}")

        if ndcg10 > best["ndcg@10"]:
            best["ndcg@10"] = ndcg10
            best["epoch"] = epoch
            best["val_metrics"] = val_metrics
            # Save weights for the best validation epoch so test metrics are comparable.
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Evaluate best model on test.
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    test_metrics = evaluate_topn_ndcg_onepos(
        model,
        device=device,
        test_negs_df=test_negs_df,
        ks=ks,
        num_movies=num_items,
    )

    out_dir = RESULTS_DIR / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "dataset": args.dataset,
        "seed": args.seed,
        "hyperparams": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "neg_ratio": args.neg_ratio,
            "embed_dim": args.embed_dim,
            "mlp_layers": mlp_layers,
            "dropout": args.dropout,
            "positive_threshold": args.positive_threshold,
            "ks": ks,
        },
        "train_positive_rows": int(len(train_pos_df)),
        "best_epoch_val_ndcg@10": best["epoch"],
        "val_metrics": best["val_metrics"],
        "test_metrics": test_metrics,
        "ranking_eval": {"val": val_eval_summary, "test": test_eval_summary},
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"[done] wrote NeuMF results to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
