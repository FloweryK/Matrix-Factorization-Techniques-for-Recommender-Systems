import numpy as np


class Evaluator:
    def __init__(self):
        self.predicts = []
        self.labels = []

    def append(self, label, predict):
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