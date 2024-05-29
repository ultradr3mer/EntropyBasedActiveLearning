from random import Random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2, InterpolationMode
from torchvision.transforms.functional import rotate
# from util import normalize
from torch.nn.functional import normalize

import plotFunctions

flip_indices = [1, 0, 2]


class LearnerDataset(Dataset):
    def __init__(self, data):
        data = np.array([np.concatenate((self.normalize_x(d[0]), [d[2] - 1.0], [self.normalize_y(d[1])]), 0)
                         for d in data] +
                        [np.concatenate((self.normalize_x(d[0][flip_indices]), [1.0 - d[2]], [self.normalize_y(d[1])]), 0)
                         for d in data])

        self.data = torch.tensor(data, dtype=torch.float)

        self.transform = v2.Compose([v2.RandomHorizontalFlip(p=0.5),
                                     v2.RandomVerticalFlip(p=0.5),
                                     v2.RandomResizedCrop(size=(64, 64), scale=(0.8, 1),
                                                          interpolation=InterpolationMode.BILINEAR)])

        self.inflate = 2
        self.data_length = len(self.data)
        self.length = self.data_length * self.inflate
        self.rnd = Random('423fsdf')

    def normalize_y(self, y):
        y -= np.average(y)
        y /= np.std(y)
        # normalize arround zero and map logarithmic
        # y = np.log(1.0 + np.abs(y)) * np.sign(y)
        # y = normalize(y)
        return y

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        d = self.data[idx % self.data_length]
        degrees = self.rnd.choice([0, 90, 180, 270])
        d = rotate(d, degrees)
        d = self.transform(d)
        return d[:-1], d[-1:]

    def normalize_x(self, x):
        x /= np.std(x)
        return x
