import numpy as np
import torch
from torch.utils.data import Dataset
import plotFunctions

class LearnerDataset(Dataset):
    def __init__(self, data):
        # if not flattern:
        #     x = np.array([np.array(d[0]).reshape(-1) for d in data])
        #     y = np.array([np.array(d[1]).reshape(-1) for d in data])
        # else:
        x = np.array([np.concatenate((d[0], d[2]),0) for d in data])
        # x = np.array([d[0] for d in data])
        y = np.array([[d[1][2]] for d in data])

        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)

        # self.transform = transforms.Compose([transforms.RandomAffine(degrees=0,
        #                                                              translate=(0.1, 0.1),
        #                                                              scale=(0.8, 1.2)),
        #                                      transforms.RandomHorizontalFlip(p=0.5),
        #                                      transforms.RandomVerticalFlip(p=0.5)])
        # img = self.x[1]
        # cv2.imwrite("testa.png", img[2].numpy() * 2**8)
        # img = self.transform(img)
        # cv2.imwrite("testb.png", img[2].numpy() * 2**8)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = self.x[idx]
        return sample, self.y[idx]