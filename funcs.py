import numpy as np
from torch.utils.data import DataLoader, random_split


def make_ui_matrix(dataset):
    n_user = dataset.n_user
    n_movie = dataset.n_movie
    ui_mat = np.zeros((n_user, n_movie))

    for data in dataset:
        user_idx, movie_idx = data['x']
        rating = data['r']
        ui_mat[user_idx][movie_idx] = rating

    return ui_mat


def calculate_rmse(labels, predicts):
    labels = np.array(labels)
    predicts = np.array(predicts)

    return np.sqrt(np.mean((labels - predicts)**2))


def split_dataset(dataset, n_batch):
    n_train = int(len(dataset) * 0.8)
    n_test = int(len(dataset) * 0.1)
    n_vali = len(dataset) - n_train - n_test
    trainset, testset, valiset = random_split(dataset, [n_train, n_test, n_vali])
    trainloader = DataLoader(dataset=trainset, batch_size=n_batch, num_workers=4, shuffle=True)
    testloader = DataLoader(dataset=testset, batch_size=n_batch, num_workers=4, shuffle=True)
    valiloader = DataLoader(dataset=valiset, batch_size=n_batch, num_workers=4, shuffle=True)
    return trainloader, testloader, valiloader