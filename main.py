import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Embedding
from config import *
from dataset import RatingDataset
from trainer import Trainer
from dataloader import read_movielens
pd.options.mode.chained_assignment = None


if __name__ == '__main__':
    # prepare prerequisites
    df = read_movielens()

    # time split train, test, vali
    t_split = df['timestamp'].tolist()[int(len(df) * SPLIT_RATIO)]
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
    user_idx2id = {idx: id for id, idx in user_id2idx.items()}
    item_idx2id = {idx: id for id, idx in item_id2idx.items()}

    # add idx column
    df_train['userIdx'] = df_train['userId'].map(lambda x: user_id2idx[x])
    df_train['itemIdx'] = df_train['itemId'].map(lambda x: item_id2idx[x])
    df_test['userIdx'] = df_test['userId'].map(lambda x: user_id2idx[x])
    df_test['itemIdx'] = df_test['itemId'].map(lambda x: item_id2idx[x])
    df_vali['userIdx'] = df_vali['userId'].map(lambda x: user_id2idx[x])
    df_vali['itemIdx'] = df_vali['itemId'].map(lambda x: item_id2idx[x])

    # make dataset
    trainset = RatingDataset(df_train)
    testset = RatingDataset(df_test)
    valiset = RatingDataset(df_vali)

    # make dataloader
    trainloader = DataLoader(dataset=trainset, batch_size=N_BATCH, shuffle=True)
    testloader = DataLoader(dataset=testset, batch_size=N_BATCH, shuffle=True)
    valiloader = DataLoader(dataset=valiset, batch_size=N_BATCH, shuffle=True)

    # model
    model = Embedding(n_user=len(user_ids), n_item=len(item_ids), n_embed=N_EMBED)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer
    )

    for epoch in range(N_EPOCH):
        print(epoch)
        trainer.single_epoch(trainloader, tag='train', epoch=epoch)
        trainer.single_epoch(testloader, tag='test', epoch=epoch)
        trainer.single_epoch(valiloader, tag='vali', epoch=epoch)

