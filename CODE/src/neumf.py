import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


def _pick_directml_device(torch_directml):
    integrated = ("radeon(tm) graphics", "intel(r) uhd", "intel(r) iris", "intel(r) hd")
    for i in range(torch_directml.device_count()):
        name = torch_directml.device_name(i)
        print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    DirectML device {i}: {name}")
        if not any(p in name.lower() for p in integrated):
            return torch_directml.device(i), name
    # fallback: just use device 0
    return torch_directml.device(0), torch_directml.device_name(0)


def _select_rocm_device():
    integrated = ("radeon(tm) graphics", "intel", "vega 8", "vega 6", "vega 3")
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        if not any(p in name.lower() for p in integrated):
            return i, name
    return 0, torch.cuda.get_device_name(0)


def _select_device():
    if torch.cuda.is_available():
        if torch.version.hip:
            idx, name = _select_rocm_device()
        else:
            idx, name = 0, torch.cuda.get_device_name(0)
        dev = torch.device(f"cuda:{idx}")
        backend = "ROCm" if torch.version.hip else "CUDA"
        print(f"  NeuMF: using GPU ({backend} - {name})")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("  NeuMF: using GPU (MPS - Apple Silicon)")
    else:
        try:
            import torch_directml
            dev, name = _pick_directml_device(torch_directml)
            print(f"  NeuMF: using GPU (DirectML - {name})")
        except ImportError:
            dev = torch.device("cpu")
            print("  NeuMF: using CPU")
    return dev

class NegativeSamplingDataset(Dataset):
    def __init__(self, df, n_items, num_negatives=4, seed=42):
        rng = np.random.default_rng(seed)
        user_pos = df.groupby("user_idx")["item_idx"].apply(set).to_dict()

        pos_u = df["user_idx"].values.astype(np.int64)
        pos_i = df["item_idx"].values.astype(np.int64)
        n = len(pos_u)

        neg_u = np.repeat(pos_u, num_negatives)
        neg_i = rng.integers(0, n_items, size=n * num_negatives)

        # Fix the small fraction of sampled negatives that collide with positives
        for idx in range(len(neg_u)):
            while neg_i[idx] in user_pos[neg_u[idx]]:
                neg_i[idx] = rng.integers(0, n_items)

        all_u = np.concatenate([pos_u, neg_u])
        all_i = np.concatenate([pos_i, neg_i])
        all_l = np.concatenate([np.ones(n, dtype=np.float32),
                                 np.zeros(n * num_negatives, dtype=np.float32)])

        perm = rng.permutation(len(all_u))
        self.users  = torch.from_numpy(all_u[perm])
        self.items  = torch.from_numpy(all_i[perm])
        self.labels = torch.from_numpy(all_l[perm])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]


class NeuMFNet(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=32, layers=[64, 32, 16]):
        super().__init__()
        # GMF embeddings (dot product)
        self.gmf_user = nn.Embedding(n_users, emb_dim)
        self.gmf_item = nn.Embedding(n_items, emb_dim)
        # MLP embeddings (separate weights)
        self.mlp_user = nn.Embedding(n_users, emb_dim)
        self.mlp_item = nn.Embedding(n_items, emb_dim)


        mlp = []
        in_dim = emb_dim * 2
        for out_dim in layers:
            mlp += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(0.2)]
            in_dim = out_dim
        
        self.mlp = nn.Sequential(*mlp)

        self.output = nn.Linear(emb_dim + layers[-1], 1)

    def forward(self, u, i):
        gmf = self.gmf_user(u) * self.gmf_item(i)  # element-wise product
        mlp_x = torch.cat([self.mlp_user(u), self.mlp_item(i)], dim=-1)
        mlp_out = self.mlp(mlp_x)
        return self.output(torch.cat([gmf, mlp_out], dim=-1)).squeeze()


class NeuMFModel:
    rating_model = False  # outputs ranking scores, not rating predictions

    def __init__(self, emb_dim=64, layers=[128,64,32], epochs=20, lr=0.001,
                 batch_size=4096, num_negatives=4):
        self.emb_dim       = emb_dim
        self.layers        = layers
        self.epochs        = epochs
        self.lr            = lr
        self.batch_size    = batch_size
        self.num_negatives = num_negatives
        self.device        = _select_device()

    def fit(self, train):
        self.n_users = train["user_idx"].max() + 1
        self.n_items = train["item_idx"].max() + 1

        print(f"  Building negative sampling dataset ({self.num_negatives} negatives per positive)...")
        dataset = NegativeSamplingDataset(train, self.n_items, self.num_negatives)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.net = NeuMFNet(self.n_users, self.n_items, self.emb_dim, self.layers).to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        criterion = nn.BCEWithLogitsLoss()

        self.net.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for u, i, label in loader:
                u, i, label = u.to(self.device), i.to(self.device), label.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.net(u, i), label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1}/{self.epochs} - Loss: {total_loss/len(loader):.4f} - LR: {current_lr:.6f}")
            scheduler.step()

    def predict(self, df):
        self.net.eval()
        u = torch.tensor(df["user_idx"].values, dtype=torch.long).to(self.device)
        i = torch.tensor(df["item_idx"].values, dtype=torch.long).to(self.device)
        u = u.clamp(0, self.n_users - 1)
        i = i.clamp(0, self.n_items - 1)
        with torch.no_grad():
            scores = torch.sigmoid(self.net(u, i)).detach().cpu().numpy()
        return scores

    def explain(self, user_idx, recommended_item_idxs, train, movies, top_k=3):
        """
        For each recommended item, return the top_k items from the user's rated
        history that are most similar to it in GMF embedding space.
        Returns a list of explanation strings, one per recommended item.
        """
        self.net.eval()

        # Items this user has rated, sorted by rating descending
        user_ratings = (train[train["user_idx"] == user_idx]
                        .sort_values("rating", ascending=False))
        rated_idxs  = user_ratings["item_idx"].values.astype(int)
        rated_scores = user_ratings["rating"].values

        if len(rated_idxs) == 0:
            return ["No rating history to explain from." for _ in recommended_item_idxs]

        item_to_title = (movies.drop_duplicates("movieId")
                               .set_index("movieId")["title"].to_dict())
        idx_to_movieid = (train.drop_duplicates("item_idx")
                               .set_index("item_idx")["movieId"].to_dict())

        with torch.no_grad():
            # GMF item embeddings encode latent item similarity
            rec_tensor   = torch.tensor(recommended_item_idxs, dtype=torch.long).to(self.device)
            rated_tensor = torch.tensor(rated_idxs, dtype=torch.long).to(self.device)
            rec_emb   = self.net.gmf_item(rec_tensor)    # (n_rec, emb_dim)
            rated_emb = self.net.gmf_item(rated_tensor)  # (n_rated, emb_dim)

            # Cosine similarity: (n_rec, n_rated)
            rec_norm   = rec_emb   / (rec_emb.norm(dim=1, keepdim=True)   + 1e-8)
            rated_norm = rated_emb / (rated_emb.norm(dim=1, keepdim=True) + 1e-8)
            sims = (rec_norm @ rated_norm.T).cpu().numpy()

        # Build explanations, tracking used items so each recommendation gets distinct reasons
        explanations = []
        used_indices = set()
        for sim_row in sims:
            ranked = np.argsort(sim_row)[::-1]
            reasons = []
            for j in ranked:
                if j in used_indices:
                    continue
                movie_id = idx_to_movieid.get(int(rated_idxs[j]))
                title    = item_to_title.get(movie_id, f"MovieID {movie_id}")
                rating   = rated_scores[j]
                reasons.append((title, rating))
                used_indices.add(j)
                if len(reasons) == top_k:
                    break
            explanations.append(reasons)

        return explanations