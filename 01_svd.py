import numpy as np
import config
from funcs import make_ui_matrix
from dataset import MovieLensDataset


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

