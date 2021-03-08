import pandas as pd
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from funcs import split_dataset_by_time, make_matrix
from models import AutoEncoder
from losses import MaskedMSELoss


class Trainer:
    def __init__(self, model):
        self.model = model
        self.criterion = MaskedMSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.writer = SummaryWriter()

    def single_epoch(self, mat_train, epoch, refeed=False):
        running_loss = 0

        for user_id in mat_train.index:
            # inference
            x = torch.tensor(mat_train.loc[user_id].tolist())
            fx = model(x)

            # calculate loss
            loss = self.criterion(fx, x)
            running_loss += loss.item()

            # update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # refeeding
            if refeed:
                ffx = model(fx.detach())
                loss = self.criterion(ffx, fx.detach())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        running_loss *= 1 / len(mat_train.index)

        # write on tensorboard
        self.writer.add_scalar(f'Loss/train', running_loss, epoch)

    def single_epoch_eval(self, mat_train, mat_test, tag, epoch):
        running_loss = 0

        for user_id in mat_test.index:
            # inference
            x = torch.tensor(mat_test.loc[user_id].tolist())
            fx = model(torch.tensor(mat_train.loc[user_id].tolist()))

            # calculate loss
            loss = self.criterion(fx, x)
            running_loss += loss.item()

        running_loss *= 1 / len(mat_test.index)

        # write on tensorboard
        self.writer.add_scalar(f'Loss/{tag}', running_loss, epoch)


if __name__ == '__main__':
    # parameters
    N_EPOCH = 200

    # load excel
    df = pd.read_csv('data/ml-latest-small/ratings.csv')
    user_ids = df['userId'].unique()
    movie_ids = df['movieId'].unique()

    # split data by time
    df_train, df_test, df_vali = split_dataset_by_time(df, ratio=0.5)
    print(f'train | data: {len(df_train)}, user: {len(df_train["userId"].unique())} movie: {len(df_train["movieId"].unique())}')
    print(f'test | data: {len(df_test)}, user: {len(df_test["userId"].unique())} movie: {len(df_test["movieId"].unique())}')
    print(f'vali | data: {len(df_vali)}, user: {len(df_vali["userId"].unique())} movie: {len(df_vali["movieId"].unique())}')

    # make data as matrix
    mat_train = make_matrix(df_train, movie_ids)
    mat_test = make_matrix(df_test, movie_ids)
    mat_vali = make_matrix(df_vali, movie_ids)
    print(mat_train)
    print(mat_test)
    print(mat_vali)

    # model, criterion, optimizer
    model = AutoEncoder(n_input=len(movie_ids),
                        n_hidden1=64,
                        n_hidden2=16,
                        n_hidden3=4)

    # train
    trainer = Trainer(model)

    for epoch in range(N_EPOCH):
        print(epoch)
        trainer.single_epoch(mat_train, epoch)
        trainer.single_epoch_eval(mat_train, mat_test, 'test', epoch)
        trainer.single_epoch_eval(mat_train, mat_vali, 'vali', epoch)






