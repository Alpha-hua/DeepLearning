import numpy as np
import torch
from torch.utils.data import Dataset

class myDataSet(Dataset):
    def __init__(self, X, Y = None):
        self.data = torch.from_numpy(X).float()
        if Y is not None:
            self.label = torch.LongTensor(Y.astype('int32'))
        else:
            self.label = None

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)