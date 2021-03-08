import numpy as np
from dataset import MovieLensDataset


def make_ui_matrix(dataset):
    n_user = dataset.n_user
    n_movie = dataset.n_movie
    ui_mat = np.zeros((n_user, n_movie))

    for data in dataset:
        user_idx, movie_idx = data['x']
        rating = data['r']
        ui_mat[user_idx][movie_idx] = rating

    return ui_mat


def run():
    dataset = MovieLensDataset(path=config.DATA_PATH, is_tensor=False)

    # make user-item matrix
    print('making user-item matrix')
    ui_mat = make_ui_matrix(dataset)

    # SVD
    print('making SVD matrices')
    U, S, Vt = np.linalg.svd(ui_mat)
    print(U.shape)
    print(S.shape)
    print(Vt.shape)


if __name__ == '__main__':
    run()

