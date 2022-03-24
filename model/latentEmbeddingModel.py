import torch
import torch.nn as nn


class LatentEmbeddingModel(nn.Module):
    def __init__(self, n_user, n_item, n_embed):
        super(LatentEmbeddingModel, self).__init__()

        # embedding layer (B, 2) -> (B, 2, N_EMBED)
        self.embedding_user = nn.Embedding(n_user, n_embed)
        self.embedding_item = nn.Embedding(n_item, n_embed)

        # global avg, bias
        self.mu = nn.Embedding(1, 1)
        self.b_user = nn.Embedding(n_user, 1)
        self.b_item = nn.Embedding(n_item, 1)

    def forward(self, x):
        # x (B, 2)
        x_user = self.embedding_user(x[:, 0].long())                # (B, N_EMBED)
        x_item = self.embedding_item(x[:, 1].long())                # (B, N_EMBED)

        mu = self.mu(torch.tensor([0]*x.shape[0])).view(-1)         # (B)
        b_user = self.b_user(x[:, 0].long()).view(-1)               # (B)
        b_item = self.b_item(x[:, 1].long()).view(-1)               # (B)

        # x = torch.sum(x_user * x_item, dim=1)  # (B)
        x = mu + b_user + b_item + torch.sum(x_user*x_item, dim=1)  # (B)
        return x