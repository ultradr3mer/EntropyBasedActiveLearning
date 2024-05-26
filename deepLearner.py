from random import Random

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

from classifier import KnnClassifier
from util import normalize

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
        self.rnd = Random('sdikjgfh')

    def pick(self, item):
        pmgr = PointManager(item)

        initial_acc = pmgr.calc_acc()

        k = 10
        points_with_index = np.concatenate((pmgr.all_x,
                                            np.array(range(len(pmgr.all_x))).reshape((len(pmgr.all_x), 1))),
                                            axis=1)
        bins = pmgr.point_bins(points_with_index)
        pred = self.predict(pmgr, False)
        i_0 = self.pick_from_pred(bins, pred, k)
        pred = self.predict(pmgr, True)
        i_1 = self.pick_from_pred(bins, pred, k)
        pick_i = set(i_0) | set(i_1)
        all_i = np.array(list(set(pmgr.labeled_i.flatten()) | pick_i), dtype=int)

        pmgr_new = PointManager(item, all_i)

        new_acc = pmgr_new.calc_acc()

        top_points = self.rnd.choices([p for b in bins for p in b],
                                    k=k)
        i_rnd = np.array(top_points, dtype=int)[:, 2]
        rnd_i = np.array(list(set(pmgr.labeled_i.flatten()) | set(i_rnd)), dtype=int)
        pmgr_rnd = PointManager(item, rnd_i)
        rnd_acc = pmgr_rnd.calc_acc()

        print(f"{new_acc-initial_acc} / {rnd_acc-initial_acc} - {new_acc-initial_acc > rnd_acc-initial_acc}")
        pass

    def pick_from_pred(self, bins, pred, k):
        pred = np.power(normalize(pred.reshape(-1)), 4)
        top_points = self.rnd.choices([p for b in bins for p in b],
                                    weights=[pred[i] for i, b in enumerate(bins) for _ in b],
                                    k=k)
        return np.array(top_points, dtype=int)[:, 2]

    def predict(self, pmgr, flip):
        prior = pmgr.calc_prior().reshape(64, 64)

        if flip:
            prior = 1.0 - prior
            tmp = pmgr.map_label_0
            pmgr.map_label_0 = pmgr.map_label_1
            pmgr.map_label_1 = tmp

        x = np.array([np.stack((pmgr.map_unlabeled,
                                pmgr.map_label_0,
                                pmgr.map_label_1,
                                prior - 0.5))])
        x = torch.tensor(x, dtype=torch.float).to(learner.device)
        pred = learner.model(x)
        return pred.detach().cpu().numpy()


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
            nn.Conv2d(4, 12, kernel_size=kernel_size, padding=padding, padding_mode='replicate'),
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

    for item in dataset:
        learner.pick(item)

