import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

from classifier import KnnClassifier

res = 32
x_resolution = res * 2
y_resolution = res * 2
model_file_name = 'model_weights.pth'


class DeepLearner:
    def __init__(self, model_file_name=None):
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {device} device")
        model = NnConv().to(device)
        if model_file_name is not None:
            model.load_state_dict(torch.load(model_file_name))
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
        padding = int(kernel_size / 2)
        self.conv_stack = nn.Sequential(
            nn.Conv2d(5, 12, kernel_size=kernel_size, padding=padding, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(12, 12, kernel_size=kernel_size, padding=padding, padding_mode='replicate'),
            # nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(12, 1, kernel_size=kernel_size, padding=padding, padding_mode='replicate'),
        )

    def forward(self, x):
        x = self.conv_stack(x)
        return x


if __name__ == '__main__':
    from generator import Generator
    from pointManager import PointManager

    # from deepTraining import DeepTrainer
    # t = DeepTrainer()
    # t.train()
    learner = DeepLearner(model_file_name)
    gen = Generator()
    dataset = gen.load_or_create_dataset()

    item = dataset[0]
    pmgr = PointManager(item)
    prior = pmgr.calc_prior()
    x = np.stack((pmgr.map_unlabeled, pmgr.map_label_0, pmgr.map_label_1, cv2.split(prior)[0]))
    pred = learner.model(x)

    my_points = pmgr.remaining_x[np.where(pmgr.remaining_y == 0)]
    point_bins = pmgr.point_bins(my_points)
    top_pred = list(enumerate(pred.reshape(-1)))
    top_pred.sort(key=lambda x: x[1], reverse=True)
    top_idx = np.array([i for i, y in top_pred])
    top_points = np.array([point_bins[i][0] for i in top_idx])[:10]

    acc = pmgr.calc_acc(np.hstack(pmgr.initial_x, top_points), pmgr.initial_y)

    new_acc = [pmgr.po]
