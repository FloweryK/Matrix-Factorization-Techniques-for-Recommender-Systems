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
        x_user = self.embedding_user(x[:, 0])
        x_item = self.embedding_item(x[:, 1])
        x = torch.sum(x_user*x_item, dim=1)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, n_input, n_hidden1, n_hidden2, n_hidden3):
        super(AutoEncoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_input, n_hidden1),
            nn.SELU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.SELU(),
            nn.Linear(n_hidden2, n_hidden3),
            nn.SELU(),
            nn.Dropout(p=0.8)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(n_hidden3, n_hidden2),
            nn.SELU(),
            nn.Linear(n_hidden2, n_hidden1),
            nn.SELU(),
            nn.Linear(n_hidden1, n_input),
            nn.SELU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

