import numpy as np


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

