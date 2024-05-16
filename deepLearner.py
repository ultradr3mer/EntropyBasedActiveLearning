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
        model = NeuralNetwork().to(device)
        print(model)
        self.model = model
        self.device = device


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        inner_res = int(x_resolution ** 2 / 2)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(x_resolution ** 2 * 3, inner_res),
            nn.ReLU(),
            nn.Linear(inner_res, inner_res),
            nn.ReLU(),
            nn.Linear(inner_res, y_resolution ** 2),
            nn.Unflatten(1, torch.Size([y_resolution, y_resolution]))
        )
        # self.conv_stack = nn.Sequential(
        #     nn.Conv2d(3, 12, kernel_size=3, padding=1),  # Input channels: 1 (grayscale)
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2),
        #     nn.Conv2d(12, 12, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     # nn.MaxPool2d(3, 2),
        #     # nn.Conv2d(32, 64, kernel_size=3, padding=1),
        #     # nn.ReLU(),
        #     # nn.MaxPool2d(3, 2)
        # )
        # self.flatten = nn.Flatten()
        #
        # in_features = self.conv_stack(torch.tensor(np.ones((3, 64, 64)), dtype=torch.float32))
        #
        # self.linear_stack = nn.Sequential(
        #     nn.Linear(np.prod(in_features.shape), 512),
        #     nn.ReLU(),
        #     nn.Linear(512, y_resolution ** 2),
        #     nn.Unflatten(1, torch.Size([32, 32]))
        # )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x


class LearnerDataset(Dataset):
    def __init__(self, data_full):
        x = np.array([d[0] for d in data_full])
        y = np.array([d[1] for d in data_full])

        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)

        self.transform = transforms.Compose([transforms.RandomAffine(degrees=0,
                                                                     translate=(0.1, 0.1),
                                                                     scale=(0.8, 1.2)),
                                             transforms.RandomHorizontalFlip(p=0.5),
                                             transforms.RandomVerticalFlip(p=0.5)])
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
