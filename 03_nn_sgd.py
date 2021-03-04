import torch
import torch.nn as nn
import config
from funcs import split_dataset
from dataset import MovieLensDataset
from trainer import Trainer


class Embedding(nn.Module):
    def __init__(self, N_USER, N_ITEM, N_EMBED):
        super(Embedding, self).__init__()

        # embedding layer (B, 2) -> (B, 2, N_EMBED)
        self.embedding_user = nn.Embedding(N_USER, N_EMBED)
        self.embedding_item = nn.Embedding(N_ITEM, N_EMBED)

    def forward(self, x):
        # x (B, 2)
        x_user = self.embedding_user(x[:, 0])
        x_item = self.embedding_item(x[:, 1])
        x = torch.sum(x_user*x_item, dim=1)
        return x


if __name__ == '__main__':
    # parameters
    N_BATCH = 10
    N_EMBED = 3
    N_EPOCH = 20

    # make dataset, dataloader
    dataset = MovieLensDataset(path=config.DATA_PATH,
                               is_tensor=True)
    trainloader, testloader, valiloader = split_dataset(dataset=dataset,
                                                        n_batch=N_BATCH)

    # make model
    model = Embedding(N_USER=dataset.n_user,
                      N_ITEM=dataset.n_movie,
                      N_EMBED=N_EMBED)

    # train
    trainer = Trainer(model)
    for epoch in range(N_EPOCH):
        print(epoch)
        trainer.single_epoch(trainloader, tag='train', epoch=epoch)
        trainer.single_epoch(testloader, tag='test', epoch=epoch)
        trainer.single_epoch(valiloader, tag='vali', epoch=epoch)

