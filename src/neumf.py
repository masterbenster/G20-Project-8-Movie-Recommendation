import torch
import torch.nn as nn

class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=32, layers=[64,32,16]):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        mlp_input = emb_dim * 2
        mlp = []
        for out in layers:
            mlp += [nn.Linear(mlp_input, out), nn.ReLU()]
            mlp_input = out
        mlp.append(nn.Linear(mlp_input, 1))
        self.mlp = nn.Sequential(*mlp)

    def forward(self, u, i):
        x = torch.cat([self.user_emb(u), self.item_emb(i)], dim=-1)
        return self.mlp(x).squeeze()