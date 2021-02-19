import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import MovieLensDataset


class Embedding(nn.Module):
    def __init__(self, N_INPUT, N_EMBED):
        super(Embedding, self).__init__()

        # embedding layer (B, 2) -> (B, 2, N_EMBED)
        self.embedding = nn.Embedding(N_INPUT, N_EMBED)

    def forward(self, x):
        # x (B, 2)
        x[:, 1] += x[:, 0]
        x = self.embedding(x)                           # (B, 2, N_EMBED)
        x = torch.sum(x[:, 0, :] * x[:, 1, :], dim=1)   # (B)
        return x


if __name__ == '__main__':
    N_BATCH = 3
    N_EPOCH = 200
    N_FACTOR = 4
    lr = 0.01
    l2_lambda = 0.01

    dataset = MovieLensDataset(path='data/ml-latest-small/ratings.csv', is_tensor=True)
    n_train = int(len(dataset)*0.8)
    n_test = int(len(dataset)*0.1)
    n_vali = len(dataset) - n_train - n_test
    trainset, testset, valiset = random_split(dataset, [n_train, n_test, n_vali])
    trainloader = DataLoader(dataset=trainset, batch_size=N_BATCH)
    testloader = DataLoader(dataset=testset, batch_size=N_BATCH)
    valiloader = DataLoader(dataset=valiset, batch_size=N_BATCH)

    print('model, loss, optimizer')
    model = Embedding(N_INPUT=(dataset.n_user + dataset.n_movie), N_EMBED=N_FACTOR)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    print('training')
    for epoch in range(N_EPOCH):
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

        print('epoch: %i | RMSE: train=%.3f, test=%.3f, vali=%.3f)' % (epoch+1, np.sqrt(train_loss), np.sqrt(test_loss), np.sqrt(vali_loss)))


