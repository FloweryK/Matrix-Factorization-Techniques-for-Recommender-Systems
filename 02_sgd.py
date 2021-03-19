import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from funcs import split_dataset_by_time
from dataset import RatingsDataset, read_amazon_dataset
from evaluator import Evaluator


class Embedding:
    def __init__(self, n_user, n_item, n_embed):
        self.X_user = np.random.normal(scale=1.0 / n_embed, size=(n_user, n_embed))
        self.X_item = np.random.normal(scale=1.0 / n_embed, size=(n_item, n_embed))

    def __call__(self, x):
        return np.dot(self.X_user[x[0]], self.X_item[x[1]])


class Trainer:
    def __init__(self, model):
        self.model = model
        self.evaluator = Evaluator()
        self.writer = SummaryWriter()

    def single_epoch(self, dataset, tag, epoch):
        lr = 0.01
        r_lambda = 0.01

        t_start = time.time()

        for data in dataset:
            x = data['x']
            r = data['r']
            r_pred = self.model(x)
            self.evaluator.append(r, r_pred)

            if tag == 'train':
                i, j = x
                e = r - r_pred
                x_user = self.model.X_user[i]
                x_item = self.model.X_item[j]
                self.model.X_user[i] += lr * (e * x_item - r_lambda * x_user)
                self.model.X_item[j] += lr * (e * x_user - r_lambda * x_item)

        metric = self.evaluator.calulate('rmse')
        t_end = time.time()

        self.writer.add_scalar(f'Time/{tag}', t_end - t_start, epoch)
        self.writer.add_scalar(f'rmse/{tag}', metric, epoch)


if __name__ == '__main__':
    # parameters
    N_EMBED = 3
    N_EPOCH = 200

    # load AMAZON FASHION data
    df = read_amazon_dataset()

    # split by time
    df_train, df_test, df_vali = split_dataset_by_time(df, ratio=0.5)

    # make dataset
    user_ids = df['userId'].unique()
    item_ids = df['itemId'].unique()
    trainset = RatingsDataset(df=df_train,
                              user_ids=user_ids,
                              item_ids=item_ids,
                              is_tensor=False)
    testset = RatingsDataset(df=df_test,
                             user_ids=user_ids,
                             item_ids=item_ids,
                             is_tensor=False)
    valiset = RatingsDataset(df=df_vali,
                             user_ids=user_ids,
                             item_ids=item_ids,
                             is_tensor=False)

    # make model
    model = Embedding(n_user=len(user_ids),
                      n_item=len(item_ids),
                      n_embed=N_EMBED)

    # training
    trainer = Trainer(model=model)

    for epoch in range(N_EPOCH):
        print(epoch)
        trainer.single_epoch(trainset, tag='train', epoch=epoch)
        trainer.single_epoch(testset, tag='test', epoch=epoch)
        trainer.single_epoch(valiset, tag='vali', epoch=epoch)

