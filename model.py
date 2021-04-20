import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, n_user, n_item, n_embed):
        super(Embedding, self).__init__()

        # embedding layer (B, 2) -> (B, 2, N_EMBED)
        self.embedding_user = nn.Embedding(n_user, n_embed)
        self.embedding_item = nn.Embedding(n_item, n_embed)

    def forward(self, x):
        # x (B, 2)
        x_user = self.embedding_user(x[:, 0])       # (B, N_EMBED)
        x_item = self.embedding_item(x[:, 1])       # (B, N_EMBED)

        x = torch.sum(x_user*x_item, dim=1)         # (B)

        return x