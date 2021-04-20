from torch.utils.tensorboard import SummaryWriter
from evaluator import Evaluator


class Trainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.evaluator = Evaluator()
        self.writer = SummaryWriter()

    def single_epoch(self, dataloader, tag, epoch):
        running_loss = 0

        for data in dataloader:
            # inference
            x = data['x'].long()
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

        running_loss *= 1 / len(dataloader)
        metric = self.evaluator.calulate('rmse')

        self.writer.add_scalar(f'Loss/{tag}', running_loss, epoch)
        self.writer.add_scalar(f'rmse/{tag}', metric, epoch)
