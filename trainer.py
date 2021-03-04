import time
import torch.nn as nn
import torch.optim as optim
from evaluator import Evaluator
from torch.utils.tensorboard import SummaryWriter


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
            x = data['x']
            r = data['r']
            r_pred = self.model(x)
            loss = self.criterion(r.float(), r_pred.float())
            running_loss += loss.item()
            self.evaluator.append(r, r_pred)

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

