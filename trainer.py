from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config import *
from dataset import RatingDataset
from model.latentEmbeddingModel import LatentEmbeddingModel
from utils import Evaluator, read_movielens, split_movielens_by_time


class Trainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.evaluator = Evaluator()
        self.writer = SummaryWriter()

    def single_epoch(self, dataloader, tag, epoch):
        running_loss = 0

        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for it, data in pbar:
            # inference
            x = data['x']
            r = data['r']
            r_pred = self.model(x)

            # evaluate
            self.evaluator.append(r, r_pred)

            # calculate loss
            loss = self.criterion(r.float(), r_pred.float())
            running_loss += loss.item()
            rmse = self.evaluator.calulate('rmse')

            # update
            if tag == 'train':
                self.optimizer.zero_grad()      # clear out the gradients
                loss.backward()                 # calculate gradients
                self.optimizer.step()           # update parameters based on gradients

            # print current performance of the net on console
            pbar.set_description(f'epoch {epoch} iter {it}: {tag} loss {running_loss/len(dataloader):.5f} rmse {rmse:.5f}')

        running_loss *= 1 / len(dataloader)
        self.writer.add_scalar(f'Loss/{tag}', running_loss, epoch)
        self.writer.add_scalar(f'rmse/{tag}', rmse, epoch)


if __name__ == '__main__':
    # prepare prerequisites
    df = read_movielens(DATA_PATH)
    df_train, df_test, df_vali, n_user, n_item = split_movielens_by_time(df, SPLIT_RATIO)

    # make dataset
    trainset = RatingDataset(df_train)
    testset = RatingDataset(df_test)
    valiset = RatingDataset(df_vali)

    # make dataloader
    trainloader = DataLoader(dataset=trainset, batch_size=N_BATCH, shuffle=True)
    testloader = DataLoader(dataset=testset, batch_size=N_BATCH, shuffle=True)
    valiloader = DataLoader(dataset=valiset, batch_size=N_BATCH, shuffle=True)

    # model
    model = LatentEmbeddingModel(n_user, n_item, N_EMBED)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # trainer
    trainer = Trainer(model, criterion, optimizer)

    for epoch in range(N_EPOCH):
        trainer.single_epoch(trainloader, tag='train', epoch=epoch)
        trainer.single_epoch(testloader, tag='test', epoch=epoch)
        trainer.single_epoch(valiloader, tag='vali', epoch=epoch)

