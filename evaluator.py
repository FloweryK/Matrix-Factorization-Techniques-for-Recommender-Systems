import torch
import numpy as np


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

