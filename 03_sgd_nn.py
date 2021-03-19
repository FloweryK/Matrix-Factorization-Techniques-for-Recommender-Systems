import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from funcs import split_dataset_by_time
from models import Embedding
from dataset import RatingsDataset, read_amazon_dataset
from evaluator import Evaluator


class Trainer:
    def __init__(self, model):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.evaluator = Evaluator()
        self.writer = SummaryWriter()

    def single_epoch(self, dataloader, tag, epoch):
        t_start = time.time()
        running_loss = 0

        for data in dataloader:
            # inference
            x = data['x']
            r = data['r']
            r_pred = self.model(x)

            # evaluate
            self.evaluator.append(r, r_pred)

            # calculate loss
            loss = self.criterion(r.float(), r_pred.float())
            running_loss += loss.item()

            # update
            if tag == 'train':
                self.optimizer.zero_grad()      # clear out the gradients
                loss.backward()                 # calculate gradients
                self.optimizer.step()           # update parameters based on gradients

        running_loss *= 1 / len(dataloader.dataset)
        metric = self.evaluator.calulate('rmse')
        t_end = time.time()

        self.writer.add_scalar(f'Loss/{tag}', running_loss, epoch)
        self.writer.add_scalar(f'Time/{tag}', t_end - t_start, epoch)
        self.writer.add_scalar(f'rmse/{tag}', metric, epoch)


if __name__ == '__main__':
    # parameters
    N_EMBED = 3
    N_EPOCH = 20
    N_BATCH = 24

    # load excel
    df = read_amazon_dataset()
    df_train, df_test, df_vali = split_dataset_by_time(df, ratio=0.5)

    # make dataset
    user_ids = df['userId'].unique()
    item_ids = df['itemId'].unique()
    trainset = RatingsDataset(df=df_train,
                              user_ids=user_ids,
                              item_ids=item_ids)
    testset = RatingsDataset(df=df_test,
                             user_ids=user_ids,
                             item_ids=item_ids)
    valiset = RatingsDataset(df=df_vali,
                             user_ids=user_ids,
                             item_ids=item_ids)

    # make dataloader
    trainloader = DataLoader(dataset=trainset,
                             batch_size=N_BATCH)
    testloader = DataLoader(dataset=testset,
                            batch_size=N_BATCH)
    valiloader = DataLoader(dataset=valiset,
                            batch_size=N_BATCH)

    # make model
    model = Embedding(n_user=len(user_ids),
                      n_item=len(item_ids),
                      n_embed=N_EMBED)

    # training
    trainer = Trainer(model)

    for epoch in range(N_EPOCH):
        print(epoch)
        trainer.single_epoch(trainloader, tag='train', epoch=epoch)
        trainer.single_epoch(testloader, tag='test', epoch=epoch)
        trainer.single_epoch(valiloader, tag='vali', epoch=epoch)

