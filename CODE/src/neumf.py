import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


def _select_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"  NeuMF: using GPU (CUDA - {torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("  NeuMF: using GPU (MPS - Apple Silicon)")
    else:
        try:
            import torch_directml
            dev = torch_directml.device()
            print(f"  NeuMF: using GPU (DirectML - {torch_directml.device_name(torch_directml.default_device())})")
        except ImportError:
            dev = torch.device("cpu")
            print("  NeuMF: using CPU")
    return dev

class RatingsDataset(Dataset):
    def __init__(self, df):
        self.users  = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.items  = torch.tensor(df["item_idx"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


class NeuMFNet(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=32, layers=[64, 32, 16]):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        mlp = []
        in_dim = emb_dim * 2
        for out_dim in layers:
            mlp += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(0.2)]
            in_dim = out_dim
        mlp.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*mlp)

    def forward(self, u, i):
        x = torch.cat([self.user_emb(u), self.item_emb(i)], dim=-1)
        return self.mlp(x).squeeze()


class NeuMFModel:
    def __init__(self, emb_dim=32, layers=[64,32,16], epochs=10, lr=0.001, batch_size=1024):
        self.emb_dim    = emb_dim
        self.layers     = layers
        self.epochs     = epochs
        self.lr         = lr
        self.batch_size = batch_size
        self.device = _select_device()

    def fit(self, train):
        self.mu        = train["rating"].mean()
        self.n_users   = train["user_idx"].max() + 1
        self.n_items   = train["item_idx"].max() + 1

        self.net = NeuMFNet(self.n_users, self.n_items, self.emb_dim, self.layers).to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        loader = DataLoader(RatingsDataset(train), batch_size=self.batch_size, shuffle=True)

        self.net.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for u, i, r in loader:
                u, i, r = u.to(self.device), i.to(self.device), r.to(self.device)
                optimizer.zero_grad()
                pred = self.net(u, i)
                loss = criterion(pred, r)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"  Epoch {epoch+1}/{self.epochs} - Loss: {total_loss/len(loader):.4f}")

    def predict(self, df):
        self.net.eval()
        u = torch.tensor(df["user_idx"].values, dtype=torch.long).to(self.device)
        i = torch.tensor(df["item_idx"].values, dtype=torch.long).to(self.device)

        # Clamp to known range to avoid index errors on unseen users/items
        u = u.clamp(0, self.n_users - 1)
        i = i.clamp(0, self.n_items - 1)

        with torch.no_grad():
            preds = self.net(u, i).detach().cpu().numpy()
        return preds.clip(0.5, 5.0)