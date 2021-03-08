import torch
from torch.utils.data import Dataset


class MovieLensDataset(Dataset):
    def __init__(self, df, user_ids, movie_ids, is_tensor=True, is_groupby_user=False):
        self.df = df
        self.size = len(df)
        self.is_tensor = is_tensor
        self.is_groupby_user = is_groupby_user
        self.user_to_idx = {user_id: i for i, user_id in enumerate(sorted(user_ids))}
        self.movie_to_idx = {movie_id: i for i, movie_id in enumerate(sorted(movie_ids))}

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        row = self.df.iloc[i]
        user_idx = self.user_to_idx[row['userId']]
        movie_idx = self.movie_to_idx[row['movieId']]
        r = row['rating']

        if self.is_tensor:
            return {
                'x': torch.tensor([user_idx, movie_idx]),
                'r': torch.tensor(r)
            }
        else:
            return {
                'x': [user_idx, movie_idx],
                'r': r
            }


