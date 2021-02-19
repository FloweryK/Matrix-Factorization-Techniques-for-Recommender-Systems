import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import config
from dataset import MovieLensDataset


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
        x =  torch.sum(x_user*x_item, dim=1)
        return x


if __name__ == '__main__':
    N_BATCH = 10
    N_EPOCH = 200
    N_FACTOR = 10
    lr = 0.01
    l2_lambda = 0.01

    print('making dataset')
    dataset = MovieLensDataset(path=config.DATA_PATH, is_tensor=True)
    n_train = int(len(dataset)*0.8)
    n_test = int(len(dataset)*0.1)
    n_vali = len(dataset) - n_train - n_test
    trainset, testset, valiset = random_split(dataset, [n_train, n_test, n_vali])
    trainloader = DataLoader(dataset=trainset, batch_size=N_BATCH, num_workers=4, shuffle=True)
    testloader = DataLoader(dataset=testset, batch_size=N_BATCH, num_workers=4, shuffle=True)
    valiloader = DataLoader(dataset=valiset, batch_size=N_BATCH, num_workers=4, shuffle=True)

    print('making model, loss, optimizer')
    model = Embedding(N_USER=dataset.n_user, N_ITEM=dataset.n_movie, N_EMBED=N_FACTOR)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    print('training')
    for epoch in range(N_EPOCH):
        # measure dt / epoch
        start = time.time()

        # train
        train_loss = 0
        for i, data in enumerate(trainloader):
            x = data['x']
            r = data['r']

            r_pred = model(x)
            loss = criterion(r.float(), r_pred.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss *= 1/len(trainset)

        # test
        test_loss = 0
        for data in testloader:
            x = data['x']
            r = data['r']

            r_pred = model(x)
            loss = criterion(r.float(), r_pred.float())
            test_loss += loss.item()
        test_loss *= 1 / len(testset)

        # vali
        vali_loss = 0
        for data in valiloader:
            x = data['x']
            r = data['r']

            r_pred = model(x)
            loss = criterion(r.float(), r_pred.float())
            vali_loss += loss.item()
        vali_loss *= 1 / len(valiset)

        # measure dt
        end = time.time()

        print('epoch: %i | RMSE: train=%.3f, test=%.3f, vali=%.3f | time: %.2f s)' % (epoch+1, np.sqrt(train_loss), np.sqrt(test_loss), np.sqrt(vali_loss), end-start))


