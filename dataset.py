import torch
import pandas as pd
from torch.utils.data import Dataset


def read_movielens_dataset():
    df = pd.read_csv('data/ml-latest-small/ratings.csv')
    df.columns = ['userId', 'itemId', 'rating', 'timestamp']
    return df


def read_amazon_dataset():
    df = pd.read_csv('data/amazon-fasion/AMAZON_FASHION.csv')
    df.columns = ['itemId', 'userId', 'rating', 'timestamp']
    return df


class RatingsDataset(Dataset):
    def __init__(self, df, user_ids, item_ids, is_tensor=True, is_groupby_user=False):
        self.df = df
        self.size = len(df)
        self.is_tensor = is_tensor
        self.is_groupby_user = is_groupby_user
        self.user_to_idx = {user_id: i for i, user_id in enumerate(sorted(user_ids))}
        self.item_to_idx = {item_id: i for i, item_id in enumerate(sorted(item_ids))}

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        row = self.df.iloc[i]
        user_idx = self.user_to_idx[row['userId']]
        item_idx = self.item_to_idx[row['itemId']]
        r = row['rating']

        if self.is_tensor:
            return {
                'x': torch.tensor([user_idx, item_idx]),
                'r': torch.tensor(r)
            }
        else:
            return {
                'x': [user_idx, item_idx],
                'r': r
            }


