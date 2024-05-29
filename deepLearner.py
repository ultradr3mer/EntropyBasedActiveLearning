from random import Random

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import v2, InterpolationMode

import plotFunctions
from classifier import KnnClassifier
from util import normalize, if_then_else
from pointManager import PointManager
import torch.nn.functional as F

x_res = 64
y_res = 64

from logger import instance as logger
from sklearn.cluster import DBSCAN


class DeepLearner:
    def __init__(self, model_file_name=None):

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        logger.info(f"Using {device} device")
        model = NnConv().to(device)
        if model_file_name is not None:
            model.load_state_dict(torch.load(model_file_name))
        logger.debug(model)
        self.model = model
        self.device = device
        self.rnd = Random('sdikjgfh')

    def pick(self, item, labeled_i):
        pmgr = PointManager(item, labeled_i)

        k = 5
        points_with_index = np.concatenate((pmgr.all_x,
                                            np.array(range(len(pmgr.all_x))).reshape((len(pmgr.all_x), 1))),
                                           axis=1)
        bins = pmgr.point_bins(points_with_index)
        pred = self.predict(pmgr, False)
        pick_i = self.pick_cluster(bins, pred, k, pmgr)
        all_i = np.unique(np.hstack((pmgr.labeled_i, pick_i)))

        initial_acc = pmgr.calc_acc()

        pmgr_new = PointManager(item, all_i)

        new_acc = pmgr_new.calc_acc()

        rnd_points = self.rnd.choices([p for b in bins for p in b],
                                      k=k)
        i_rnd = np.array(rnd_points, dtype=int)[:, 2]
        rnd_i = np.array(list(set(pmgr.labeled_i.flatten()) | set(i_rnd)), dtype=int)
        pmgr_rnd = PointManager(item, rnd_i)
        rnd_acc = pmgr_rnd.calc_acc()

        success = new_acc - initial_acc > rnd_acc - initial_acc
        all_i = np.unique(np.hstack((all_i, rnd_i)))
        return all_i, success

    def pick_max(self, bins, pred, k):
        not_empty_i = [i for i, b in enumerate(bins) if len(b) > 0]
        not_empty_b = [bins[i] for i in not_empty_i]
        i = np.argmax(pred.reshape(-1)[not_empty_i])
        return np.array(self.rnd.choices(not_empty_b[i], k=k), dtype=int)[:, 2]

    def pick_cluster(self, bins, pred, k, pmgr):
        pred = pred.reshape(x_res, x_res)
        pred_sharp = pred - gaussian_filter(pred, sigma=10)
        possible_centers_map = np.array([if_then_else(len(b) > 0, 1, 0) for i, b in enumerate(bins)])
        pred_sharp = normalize(normalize(pred_sharp) * np.array(possible_centers_map).reshape(x_res, x_res))

        map = [(x, y)
               for y, row in enumerate(pred_sharp)
               for x, cell in enumerate(row)
               if cell > (0.9 - self.rnd.random() / 2)]
        centeroids = pmgr.means_set(map, k)
        return np.array([self.rnd.choice(bins[map[c][0] + map[c][1] * x_res])[2] for c in centeroids], dtype=int)

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
        x = torch.tensor(x, dtype=torch.float).to(self.device)
        pred = self.model(x)
        return pred.detach().cpu().numpy()


class NnConv(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 11
        padding = int(kernel_size / 2)
        self.conv_stack = nn.Sequential(
            nn.Conv2d(4, 12, kernel_size=kernel_size, padding=padding, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(12, 12, kernel_size=kernel_size, padding=padding, padding_mode='replicate'),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 6, kernel_size=kernel_size, padding=padding, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(6, 1, kernel_size=kernel_size, padding=padding, padding_mode='replicate'),
        )

    def forward(self, x):
        x = self.conv_stack(x)
        return x
