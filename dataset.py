import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class MovieLensDataset(Dataset):
    def __init__(self, path, is_tensor=True):
        self.path = path
        self.is_tensor = is_tensor
        self.data = []
        self.size = None
        self.n_user = None
        self.n_movie = None
        self.user_to_idx = None
        self.movie_to_idx = None

        self.load_data()

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if self.is_tensor:
            return {
                'x': torch.tensor(self.data[i]['x']),
                'r': torch.tensor(self.data[i]['r'])
            }
        else:
            return {
                'x': self.data[i]['x'],
                'r': self.data[i]['r']
            }

    def load_data(self):
        ratings = pd.read_csv(self.path)

        self.user_to_idx = {userid: idx for idx, userid in enumerate(set(ratings['userId']))}
        self.movie_to_idx = {movieid: idx for idx, movieid in enumerate(set(ratings['movieId']))}
        self.size = len(ratings.index)
        self.n_user = len(self.user_to_idx)
        self.n_movie = len(self.movie_to_idx)

        for i in range(self.size):
            userid = ratings.loc[i, 'userId']
            movieid = ratings.loc[i, 'movieId']
            rating = ratings.loc[i, 'rating']

            self.data.append({'x': [self.user_to_idx[userid], self.movie_to_idx[movieid]],
                              'r': rating})


if __name__ == '__main__':
    trainset = MovieLensDataset(path='data/ml-latest-small/ratings.csv')
    trainloader = DataLoader(dataset=trainset, batch_size=3)

    for data in trainloader:
        print(data)




