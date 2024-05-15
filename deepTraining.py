import math
import random
import json
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter

from classifier import KnnClassifier
from plotFunctions import plot_dataset, plot, plot_img
from util import if_then_else

import cv2
import os

dataset_fielname = 'dataset.json'
class_count = 12
data_range = 20
res = 32


class Generator:
    def __init__(self):
        self.rnd = random.Random('dfg43')

    def generate(self):
        center_list = []
        std_list = []
        for i in range(class_count):
            next_center = [self.rnd_float(-18, 18), self.rnd_float(-18, 18)]
            next_std = self.rnd_float(3, 5)
            if any([1 for c, std in zip(center_list, std_list)
                    if np.linalg.norm(np.array(c) - np.array(next_center)) < (next_std + std)]):
                continue
            center_list.append(next_center)
            std_list.append(next_std)

        all_x, all_y = make_blobs(n_samples=self.rnd_int(300, 600),
                                  n_features=class_count,
                                  centers=np.array(center_list),
                                  cluster_std=np.array(std_list))
        all_x = [list(p) for p in all_x]
        y_map = {i: self.rnd_int(0, 1) for i in range(class_count)}
        all_y = [y_map[y] for y in all_y]

        return (all_x, all_y)  # train_test_split(all_x, all_y, test_size=self.rnd_float(0.1, 0.9), shuffle=True)

    def rnd_float(self, min, max):
        return self.rnd.random() * (max - min) + min

    def rnd_int(self, min, max):
        return self.rnd.randint(min, max)

    def read(self, json_string):
        data = json.loads(json_string)
        for item in data:
            x_labeled, x_unlabeled, y_labeled, y_unlabeled = item
            pass


class DeepTrainer(object):
    def __init__(self, data_set):
        self.data_set = data_set
        self.clf = KnnClassifier(2, 3)
        self.counter = 0
        self.random = random.Random('sidjhf')

    def train(self):
        for i, item in enumerate(self.data_set):
            if i < 40:
                continue
            self.histogram(item, i, 'data')

    def histogram(self, item, number, folder):
        all_x, all_y = item
        all_x = np.array(all_x)
        all_y = np.array(all_y)
        classes = np.unique(all_y)
        labeled_i = self.means_set(all_x, 5)
        remaining_x = np.array([x for i, x in enumerate(all_x) if i not in labeled_i])
        remaining_y = np.array([y for i, y in enumerate(all_y) if i not in labeled_i])
        initial_x = np.array(all_x[labeled_i])
        initial_y = np.array(all_y[labeled_i])

        map_label_0 = gaussian_filter(self.histogram_class(initial_x[np.where(initial_y == 0)]), sigma=2)
        map_label_1 = gaussian_filter(self.histogram_class(initial_x[np.where(initial_y == 1)]), sigma=2)
        map_unlabeled = gaussian_filter(self.histogram_class(remaining_x), sigma=2)
        x = np.array(list(zip(map_unlabeled.reshape(-1), map_label_0.reshape(-1), map_label_1.reshape(-1)))).reshape(
            res, -1)
        x = self.normalize(x)

        self.save_as_image(x, f'{folder}/{number}_x.png')

        initial_acc = self.calc_acc(initial_x, initial_y, all_x, all_y)

        # plot_img(np.dstack((map_label_0, map_label_1, map_unlabeled)))

        # plot(remaining_x,
        #      remaining_x,
        #      remaining_y)


        def acc_for_point(p):
            acc = self.calc_acc(np.concatenate((initial_x, [p])),
                                np.concatenate((initial_y, [0])),
                                all_x,
                                all_y)
            return acc

        discovery_map = self.histogram_class([all_x[i] for i in self.means_set(all_x, 30)]).reshape(-1)
        cell_size = data_range * 2 / res
        cell_center = data_range - cell_size / 2
        x_list = np.linspace(-cell_center, cell_center, res)
        y_list = np.linspace(-cell_center, cell_center, res)
        point_coords = np.array([[x, y]
                                 for y_idx, y in enumerate(y_list)
                                 for x_idx, x in enumerate(x_list)])
        # plot_img(self.clf.predict_by_point(point_coords).reshape(res, res, 2)[:,:,0])

        map_unlabeled_flat = map_unlabeled.reshape(-1)
        discovery_points = np.array([i for i, coord in enumerate(point_coords) if map_unlabeled_flat[i] > 0])
        current_points = np.array([i for i in discovery_points if discovery_map[i] > 0.0])
        current_acc = np.array([acc_for_point(point_coords[i]) for i in current_points])
        remaining_points = [i for i in discovery_points if discovery_map[i] == 0.0]

        # acc_map = np.zeros((res, res)).reshape(-1)
        # for acc_idx, point_idx in enumerate(current_points):
        #     acc_map[point_idx] = current_acc[acc_idx]
        # plot_img(acc_map.reshape(res, res))

        discovery_hotmap = KnnClassifier(1, 3)
        for i in range(4):
            if len(remaining_points) < 10:
                return

            discovery_hotmap.fit([point_coords[i] for i in current_points], np.ones(current_acc.shape) - current_acc)

            # plot(np.array([point_coords[i] for i in current_points]),
            #      np.array([point_coords[i] for i in remaining_points]),
            #      np.array([1 for i in remaining_points]))
            #
            # pred = discovery_hotmap.predict(point_coords)
            # plot_img(pred.reshape(res, res))

            weights = self.normalize(discovery_hotmap.predict([point_coords[i]
                                                               for i in
                                                               remaining_points])) + 0.1
            if math.isnan(sum(weights)):
                weights = np.ones(len(weights))

            additional_points = self.random.choices(remaining_points,
                                                    weights=weights,
                                                    k=10)
            current_points = np.concatenate((current_points, additional_points))
            new_acc = [acc_for_point(point_coords[i]) for i in additional_points]
            current_acc = np.concatenate((current_acc, new_acc))
            remaining_points = [i for i in remaining_points if i not in current_points]

        pass

        #
        # def acc_map_cell(x_idx, x, y_idx, y):
        #     if map_unlabeled[y_idx][x_idx] == 0.0:
        #         return 0
        #     acc_for_point(np.array([x, y]))

        pred = np.zeros((res, res)).reshape(-1)
        pred = discovery_hotmap.predict(point_coords)
        # for acc_idx, point_idx in enumerate(remaining_points):
        #     pred[point_idx] = discovery_hotmap.predict([point_coords[point_idx]])
        # for i, a in zip(current_points, current_acc):
        #     pred[i] = a
        pred = np.array([max(a, initial_acc) for a in pred])
        pred = self.normalize(pred)
        # plot_img(pred.reshape(res, res))

        # plot(initial_x, remaining_x, remaining_y)

        self.save_as_image(pred.reshape(res, res), f'{folder}/{number}_y.png')

        plot_map = np.dstack((self.histogram_class(all_x[np.where(all_y == 0)]),
                              self.histogram_class(all_x[np.where(all_y == 1)]),
                              self.histogram_class(initial_x)))
        pixel = np.array([if_then_else(x > 0, 255, 0) for x in plot_map.reshape(-1)]).reshape(plot_map.shape)
        cv2.imwrite(f'{folder}/{number}_plot.png', pixel)

        pass

    def histogram_class(self, x):
        x = np.array(x)
        H, x_edges, y_edges = np.histogram2d(x[:, 0], x[:, 1], bins=res,
                                             range=[[-data_range, data_range], [-data_range, data_range]])
        H = np.log(H + 1)
        return H.T

    # def max_y_map(self, points):
    #
    #
    #
    #     rows = self.bin(1, points)
    #     cells = [[max([p[2] for p in b], default=0) for b in bin(0, r)] for r in rows]
    #     return np.array(cells).reshape(res, res)

    def normalize(self, x):
        x = x - np.min(x)
        x = x / np.max(x)
        return x

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

    def means_set(self, x, n):
        kmeans = KMeans(n_clusters=n, random_state=3456987)
        kmeans.fit(x)
        centroids = kmeans.cluster_centers_
        indices = np.array([np.argmin([np.linalg.norm(x - c) for i, x in enumerate(x)]) for c in centroids])
        return indices

    def calc_acc(self, train_x, train_y, valid_x, valid_y):
        self.clf.fit(train_x, train_y)
        self.counter = self.counter + 1
        print(self.counter)
        return self.clf.prediction_acc(valid_x, valid_y)

    def save_as_image(self, x, filename):
        image = cv2.cvtColor(np.array(x * 2 ** 16, dtype='uint16'), cv2.CV_16U)
        cv2.imwrite(filename, image)
        pass


class Main:
    def __init__(self):
        pass

    def load_or_create_dataset(self):
        try:
            with open(dataset_fielname, 'r') as f:
                json_string = f.read()
                return json.loads(json_string)
        except FileNotFoundError:
            gen = Generator()
            data_set = []
            for i in range(500):
                x, y = gen.generate()
                data_set.append([list(x), y])
            json_string = json.dumps(data_set)
            with open(dataset_fielname, 'w') as f:
                f.write(json_string)
            return data_set

    def run(self):
        data_set = self.load_or_create_dataset()
        trainer = DeepTrainer(data_set)
        trainer.train()


if __name__ == '__main__':
    main = Main()
    main.run()
