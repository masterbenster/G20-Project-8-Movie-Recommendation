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
        # GMF embeddings (dot product)
        self.gmf_user = nn.Embedding(n_users, emb_dim)
        self.gmf_item = nn.Embedding(n_items, emb_dim)
        # MLP embeddings (separate weights)
        self.mlp_user = nn.Embedding(n_users, emb_dim)
        self.mlp_item = nn.Embedding(n_items, emb_dim)


        mlp = []
        in_dim = emb_dim * 2
        for out_dim in layers:
            mlp += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(0.1)]
            in_dim = out_dim
        
        self.mlp = nn.Sequential(*mlp)

        self.output = nn.Linear(emb_dim + layers[-1], 1)

    def forward(self, u, i):
        gmf = self.gmf_user(u) * self.gmf_item(i)  # element-wise product
        mlp_x = torch.cat([self.mlp_user(u), self.mlp_item(i)], dim=-1)
        mlp_out = self.mlp(mlp_x)
        return self.output(torch.cat([gmf, mlp_out], dim=-1)).squeeze()


class NeuMFModel:
    def __init__(self, emb_dim=64, layers=[128,64,32], epochs=50, lr=0.0005, batch_size=4096):
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
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
        
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
            
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1}/{self.epochs} - Loss: {total_loss/len(loader):.4f} - LR: {current_lr:.6f}")
            scheduler.step()

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