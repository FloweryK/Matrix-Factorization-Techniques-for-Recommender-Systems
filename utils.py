
import torch
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None


def read_movielens(path):
    df = pd.read_csv(path)
    df.columns = ['userId', 'itemId', 'rating', 'timestamp']
    df = df.sort_values(by=['timestamp'])
    return df


def split_movielens_by_time(df, split_ratio):
    # time split train, test, vali
    t_split = df['timestamp'].tolist()[int(len(df) * split_ratio)]
    df_train = df[df['timestamp'] <= t_split]
    df_future = df[(df['timestamp'] > t_split)
                   & df['userId'].isin(df_train['userId'].unique())
                   & df['itemId'].isin(df_train['itemId'].unique())].sample(frac=1)
    df_test = df_future.iloc[:int(0.5 * len(df_future))]
    df_vali = df_future.iloc[int(0.5 * len(df_future)):]

    print(f'size | trainset: {len(df_train)} | testset: {len(df_test)} | valiset: {len(df_vali)}')
    print(f'unique users | trainset: {len(df_train["userId"].unique())} | testset: {len(df_test["userId"].unique())} | valiset: {len(df_vali["userId"].unique())}')
    print(f'unique items | trainset: {len(df_train["itemId"].unique())} | testset: {len(df_test["itemId"].unique())} | valiset: {len(df_vali["itemId"].unique())}')

    # make id2idx, idx2id
    user_ids = set().union(*[df_train['userId'].unique(), df_test['userId'].unique(), df_vali['userId'].unique()])
    item_ids = set().union(*[df_train['itemId'].unique(), df_test['itemId'].unique(), df_vali['itemId'].unique()])
    user_id2idx = {id: idx for idx, id in enumerate(user_ids)}
    item_id2idx = {id: idx for idx, id in enumerate(item_ids)}

    # add idx column
    df_train['userIdx'] = df_train['userId'].map(lambda x: user_id2idx[x])
    df_train['itemIdx'] = df_train['itemId'].map(lambda x: item_id2idx[x])
    df_test['userIdx'] = df_test['userId'].map(lambda x: user_id2idx[x])
    df_test['itemIdx'] = df_test['itemId'].map(lambda x: item_id2idx[x])
    df_vali['userIdx'] = df_vali['userId'].map(lambda x: user_id2idx[x])
    df_vali['itemIdx'] = df_vali['itemId'].map(lambda x: item_id2idx[x])

    return df_train, df_test, df_vali, len(user_ids), len(item_ids)


class Evaluator:
    def __init__(self):
        self.predicts = []
        self.labels = []

    def append(self, label, predict):
        if type(label) == torch.Tensor:
            if label.dim() == 0:
                self.predicts.append(predict)
                self.labels.append(label)
            elif label.dim() == 1:
                self.predicts += predict.tolist()
                self.labels += label.tolist()
        else:
            self.predicts.append(predict)
            self.labels.append(label)

    def calulate(self, kind):
        self.predicts = np.array(self.predicts)
        self.labels = np.array(self.labels)

        if kind == 'rmse':
            value = np.sqrt(np.mean((self.labels - self.predicts)**2))
        else:
            raise KeyError(f'invalid kind: {kind}')

        self.clear()
        return value

    def clear(self):
        self.predicts = []
        self.labels = []

