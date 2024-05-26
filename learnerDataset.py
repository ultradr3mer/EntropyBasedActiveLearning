import numpy as np
import torch
from torch.utils.data import Dataset
import plotFunctions


class LearnerDataset(Dataset):
    def __init__(self, data):
        x = np.array([np.concatenate((d[0], [d[2] - 0.5]), 0) for d in data])
        y = np.array([[d[1]] for d in data])

        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = self.x[idx]
        return sample, self.y[idx]
