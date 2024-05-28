import random

import numpy as np
from random import Random

from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans

from classifier import KnnClassifier

data_range = 20
res = 64

class PointManager:
    def __init__(self, item=None, labeled_i=None):
        self.clf = KnnClassifier(1, 3)
        self.rnd = Random('dfg43')
        if item is None:
            return

        self.all_x, self.all_y = item
        self.all_x = np.array(self.all_x)
        self.all_y = np.array(self.all_y)
        self.classes = np.array(range(2))

        if labeled_i is not None:
            self.labeled_i = labeled_i
        else:
            self.labeled_i = self.means_set(self.all_x, random.randint(5, 20))

        self.remaining_x = np.array([x for i, x in enumerate(self.all_x) if i not in self.labeled_i])
        self.remaining_y = np.array([y for i, y in enumerate(self.all_y) if i not in self.labeled_i])
        self.initial_x = np.array(self.all_x[self.labeled_i])
        self.initial_y = np.array(self.all_y[self.labeled_i])
        self.point_coords = self.gen_point_coords(res)

        self.map_label_0 = gaussian_filter(self.histogram_class(
            self.initial_x[np.where(self.initial_y == 0)]), sigma=2)
        self.map_label_1 = gaussian_filter(self.histogram_class(
            self.initial_x[np.where(self.initial_y == 1)]), sigma=2)
        self.map_unlabeled = gaussian_filter(self.histogram_class(self.remaining_x), sigma=2)

    def means_set(self, x, n):
        kmeans = KMeans(n_clusters=n, random_state=3456987)
        kmeans.fit(x)
        centroids = kmeans.cluster_centers_
        indices = np.array([np.argmin([np.linalg.norm(x - c) for i, x in enumerate(x)]) for c in centroids])
        return indices

    def gen_point_coords(self, res):
        cell_size = data_range * 2 / res
        cell_center = data_range - cell_size / 2
        x_list = np.linspace(-cell_center, cell_center, res)
        y_list = np.linspace(-cell_center, cell_center, res)
        point_coords = np.array([[x, y]
                                 for y_idx, y in enumerate(y_list)
                                 for x_idx, x in enumerate(x_list)])
        return point_coords

    def histogram_class(self, x, bins=res):
        x = np.array(x)
        H, x_edges, y_edges = np.histogram2d(x[:, 0], x[:, 1], bins=bins,
                                             range=[[-data_range, data_range], [-data_range, data_range]])
        H = np.log(H + 1)
        return H.T

    def acc_for_point(self, p):
        acc0 = self.calc_acc_for(np.concatenate((self.initial_x, [p])),
                                 np.concatenate((self.initial_y, [0])))
        acc1 = self.calc_acc_for(np.concatenate((self.initial_x, [p])),
                                 np.concatenate((self.initial_y, [1])))
        return [acc0, acc1]

    def calc_acc_for(self, x_train, y_train):
        self.clf.fit(x_train, y_train)
        return self.clf.prediction_acc(self.all_x, self.all_y)

    def calc_acc(self):
        self.clf.fit(self.initial_x, self.initial_y)
        return self.clf.prediction_acc(self.all_x, self.all_y)

    def calc_prior(self):
        self.clf.fit(self.initial_x, self.initial_y)
        prior = self.clf.predict_by_point(self.point_coords).reshape(res, res, -1)
        return prior

    def bin(self, axis, points):
        thresholds = np.linspace(-20, 20, res + 1)
        bins = []
        sorted = list(points)
        sorted.sort(key=lambda x: -x[axis])
        for x in thresholds[1:]:
            cur_bin = []
            while len(sorted) > 0 and sorted[-1][axis] < x:
                cur_bin.append(sorted.pop())
            bins.append(cur_bin)
        return bins

    def point_bins(self, points):
        return [np.array(b) for row in self.bin(1, points)
                for b in self.bin(0, row)]
