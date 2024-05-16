import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

res = 32
x_resolution = res * 2
y_resolution = res


class DeepLearner:
    def __init__(self):
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {device} device")
        model = NnConv().to(device)
        print(model)
        self.model = model
        self.device = device


class NnFully(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten(1)
        inner_res = x_resolution ** 2
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(x_resolution ** 2 * 3, inner_res),
            nn.ReLU(),
            nn.Linear(inner_res, inner_res),
            nn.ReLU(),
            nn.Linear(inner_res, y_resolution ** 2),
            nn.Unflatten(1, torch.Size([1, y_resolution, y_resolution]))
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x


class NnConv(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 9
        padding = int((kernel_size / 2))
        self.conv_stack = nn.Sequential(
            nn.Conv2d(5, 12, kernel_size=kernel_size, padding=padding, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(12, 12, kernel_size=kernel_size, padding=padding, padding_mode='replicate'),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(12, 1, kernel_size=kernel_size, padding=padding, padding_mode='replicate'),
        )

    def forward(self, x):
        x = self.conv_stack(x)
        return x


class LearnerDataset(Dataset):
    def __init__(self, data):
        # if not flattern:
        #     x = np.array([np.array(d[0]).reshape(-1) for d in data])
        #     y = np.array([np.array(d[1]).reshape(-1) for d in data])
        # else:
        x = np.array([np.concatenate((d[0], d[2][:2]),0) for d in data])
        # x = np.array([d[0] for d in data])
        y = np.array([[d[1]] for d in data])

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


if __name__ == '__main__':
    from deepTraining import Main

    main = Main()
    main.run()
