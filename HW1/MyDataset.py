from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np


class MyDataset(Dataset):
    def __init__(self, filepath):
        self.dataset = pd.read_csv(filepath)
        self.dataset = np.array(self.dataset.values.tolist())
        self.data = torch.from_numpy(np.float32(self.dataset[:, 0:-1]))
        self.label = torch.from_numpy(np.float32(self.dataset[:, [-1]]))

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

