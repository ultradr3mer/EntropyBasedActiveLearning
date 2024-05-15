import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

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
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(x_resolution ** 2, 512, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(512, 512, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(512, y_resolution ** 2, dtype=torch.float32)
        )

    def forward(self, x):
        x = self.flatten(x)
        predict = self.linear_relu_stack(x)
        return predict


class LearnerDataset(Dataset):
    def __init__(self, data_full):
        x = np.array([np.array(d[0]).reshape(-1) for d in data_full])
        y = np.array([np.array(d[1]).reshape(-1) for d in data_full])

        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = self.x[idx]
        return sample, self.y[idx]

if __name__ == '__main__':
    from deepTraining import Main
    main = Main()
    main.run()