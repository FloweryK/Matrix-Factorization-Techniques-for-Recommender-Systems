import numpy as np
import config
from funcs import make_ui_matrix, calculate_rmse
from dataset import MovieLensDataset
from evaluator import Evaluator


def run():
    factors = 3
    epochs = 200
    lr = 0.01
    r_lambda = 0.01

    dataset = MovieLensDataset(path=config.DATA_PATH, is_tensor=False)
    evaluator = Evaluator()

    # Initialize Latent Factor
    n_user, n_movie = dataset.n_user, dataset.n_movie
    X_user = np.random.normal(scale=1.0 / factors, size=(n_user, factors))
    X_movie = np.random.normal(scale=1.0 / factors, size=(n_movie, factors))

    # SGD
    print('optimizing in SGD')
    for epoch in range(epochs):
        for data in dataset:
            i, j = data['x']
            rating = data['r']

            # update
            predict = np.dot(X_user[i], X_movie[j])
            error = rating - predict
            X_user[i] += lr * (error * X_movie[j] - r_lambda * X_user[i])
            X_movie[j] += lr * (error * X_user[i] - r_lambda * X_movie[j])

            evaluator.append(rating, predict)

        if epoch % 10 == 0:
            rmse = evaluator.calulate('rmse')
            print("iter step: {0}, rmse: {1:4f}".format(epoch, rmse))


if __name__ == '__main__':
    run()

