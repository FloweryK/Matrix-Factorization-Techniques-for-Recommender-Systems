import torch
from torch.utils.data import Dataset


class RatingDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.size = len(df)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        row = self.df.iloc[i]
        user_idx = row['userIdx']
        item_idx = row['itemIdx']
        r = row['rating']

        return {
            'x': torch.tensor([user_idx, item_idx]),
            'r': torch.tensor(r)
        }
