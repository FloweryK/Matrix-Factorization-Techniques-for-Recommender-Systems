import numpy as np
from funcs import make_ui_matrix, calculate_rmse
from dataset import MovieLensDataset


def run():
    factors = 3
    epochs = 200
    lr = 0.01
    r_lambda = 0.01

    dataset = MovieLensDataset(path='data/ml-latest-small/ratings.csv', is_tensor=False)

    # make user-item matrix
    print('making user-item matrix')
    ui_mat = make_ui_matrix(dataset)

    # SGD
    print('optimizing in SGD')
    n_user, n_movie = dataset.n_user, dataset.n_movie
    X_user = np.random.normal(scale=1.0 / factors, size=(n_user, factors))
    X_movie = np.random.normal(scale=1.0 / factors, size=(n_movie, factors))
    non_zeros = [(i, j, ui_mat[i, j]) for i in range(n_user) for j in range(n_movie) if ui_mat[i, j] > 0]

    for epoch in range(epochs):
        for i, j, r in non_zeros:
            # update
            e = r - np.dot(X_user[i], X_movie[j])
            X_user[i] += lr * (e * X_movie[j] - r_lambda * X_user[i])
            X_movie[j] += lr * (e * X_user[i] - r_lambda * X_movie[j])

        # evaluate
        labels = [rating for _, _, rating in non_zeros]
        predicts = [np.dot(X_user[i], X_movie[j]) for i, j, rating in non_zeros]

        rmse = calculate_rmse(labels, predicts)
        if epoch % 10 == 0:
            print("iter step: {0}, rmse: {1:4f}".format(epoch, rmse))


if __name__ == '__main__':
    run()

