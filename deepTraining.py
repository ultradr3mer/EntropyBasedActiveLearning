import math
import os
import random
import json
import numpy as np
import torch
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter
from torch import nn, optim
from torch.utils.data import DataLoader

from classifier import KnnClassifier
from deepLearner import DeepLearner, x_resolution, y_resolution, LearnerDataset
from plotFunctions import plot_img
from util import if_then_else

import cv2

dataset_fielname = 'dataset.json'
class_count = 12
data_range = 20
uint16_max = 2 ** 16

res = x_resolution


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
    def __init__(self, data_set, learner):
        self.data_set = data_set
        self.clf = KnnClassifier(2, 3)
        self.counter = 0
        self.random = random.Random('sidjhf')
        self.learner = learner

    def join_channels(self, x):
        target_res = int(x.shape[0] / 2)
        row_merge = np.rot90(x[:, :, 0].reshape(target_res * 2, target_res, 2))
        return np.rot90(row_merge.reshape(target_res, target_res, 4), k=-1)[:, :, :3]

    def load_or_gen_train_data(self, dataset, number):
        folder = 'data'
        x_filename = f'{folder}/{number}_x.png'
        y_filename = f'{folder}/{number}_y.png'
        try:
            x = self.load_from_image(x_filename)[:3]
            y = self.load_from_image(y_filename)[0]
            if x.shape != (3, 64, 64) or y.shape != (32, 32):
                os.remove(x_filename)
                os.remove(y_filename)
                raise FileNotFoundError()
            return x, y
        except FileNotFoundError:
            x, y = self.gen_training_histogram(dataset)
            self.save_as_image(x, x_filename)
            self.save_as_image(y, y_filename)
            return np.array(cv2.split(x)), y

    def train_step(self, dataloader, model, loss_fn, optimizer):
        device = self.learner.device
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(self, dataloader, model, loss_fn):
        device = self.learner.device
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss = 0  # , correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        # correct /= size
        print(f"Test Error: Avg loss: {test_loss:>8f} \n")

    def train(self):
        model = self.learner.model
        indexed_data = list(enumerate(self.data_set))
        train, test = train_test_split(indexed_data, test_size=0.1, shuffle=True)
        train_dataloader = self.setup_dataloader(train)
        test_dataloader = self.setup_dataloader(test)

        for X, y in test_dataloader:
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
        epochs = 50
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            scheduler.step(t)
            self.train_step(train_dataloader, model, loss_fn, optimizer)
            self.test(test_dataloader, model, loss_fn)
        print("Done!")

        #
        # for i, item in enumerate(self.data_set):
        #     # if i < 40:
        #     #     continue
        #     x, y = self.load_or_gen_train_data(item, i)
        #
        #     batch_size = 64

        # Create data loaders.

    def setup_dataloader(self, data):
        data_full = list([self.load_or_gen_train_data(item, i) for i, item in data])
        my_dataset = LearnerDataset(data_full)
        return DataLoader(my_dataset, batch_size=32, shuffle=True)

    def sainty_check(self, item, y):
        all_x, all_y = item
        all_x = np.array(all_x)
        all_y = np.array(all_y)
        classes = np.unique(all_y)
        labeled_i = self.means_set(all_x, 5)
        remaining_x = np.array([x for i, x in enumerate(all_x) if i not in labeled_i])
        remaining_y = np.array([y for i, y in enumerate(all_y) if i not in labeled_i])
        initial_x = np.array(all_x[labeled_i])
        initial_y = np.array(all_y[labeled_i])

        initial_acc = self.calc_acc(initial_x, initial_y, all_x, all_y)

        point_coords = self.gen_point_coords(res)

        point_bins = [b for row in self.bin(1, remaining_x[np.where(remaining_y == 0)]) for b in self.bin(0, row)]
        top_y = list(enumerate(y.reshape(-1)))
        top_y.sort(key=lambda x: x[1], reverse=True)
        top_idx = np.array([i for i, y in top_y])
        top_points = np.array([p for i in top_idx for p in point_bins[i]])[:100]

        # for

        def acc_for_point(p):
            acc = self.calc_acc(np.concatenate((initial_x, [p])),
                                np.concatenate((initial_y, [0])),
                                all_x,
                                all_y)
            return acc

        acc = [acc_for_point(p) - initial_acc for p in top_points]

        pass

    def gen_training_histogram(self, item):
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
        x = np.dstack((map_unlabeled, map_label_0, map_label_1))
        # x = np.array(list(zip(map_unlabeled.reshape(-1), map_label_0.reshape(-1), map_label_1.reshape(-1)))).reshape(
        #     res, -1)
        x = self.normalize(x)

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

        point_coords = self.gen_point_coords(res)
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

        y = np.zeros((y_resolution, y_resolution)).reshape(-1)
        y = discovery_hotmap.predict(self.gen_point_coords(y_resolution)).reshape(y_resolution, y_resolution)
        map_unlabeled_low = gaussian_filter(self.histogram_class(remaining_x, y_resolution), sigma=1)
        density_hard = np.array(self.contrast(self.normalize(map_unlabeled_low), 0.2, 0.3))
        y = self.normalize(np.array([max(a, initial_acc) for a in y.reshape(-1)])).reshape(y_resolution, y_resolution)
        y = y * density_hard
        # plot_img(y)

        # plot(initial_x, remaining_x, remaining_y)

        # plot_map = np.dstack((self.histogram_class(all_x[np.where(all_y == 0)]),
        #                       self.histogram_class(all_x[np.where(all_y == 1)]),
        #                       self.histogram_class(initial_x)))
        # pixel = np.array([if_then_else(x > 0, 255, 0) for x in plot_map.reshape(-1)]).reshape(plot_map.shape)
        # cv2.imwrite(f'{folder}/{number}_plot.png', pixel)

        return x, y

    def histogram_class(self, x, bins=res):
        x = np.array(x)
        H, x_edges, y_edges = np.histogram2d(x[:, 0], x[:, 1], bins=bins,
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
        image = cv2.cvtColor(np.array(x * uint16_max, dtype='uint16'), cv2.CV_16U)
        cv2.imwrite(filename, image)
        pass

    def load_from_image(self, filename):
        img = cv2.imread(filename, -1)
        if img is None:
            raise FileNotFoundError(f"Image with {filename} could not be loaded.")
        return np.array(cv2.split(img), dtype=float) / uint16_max

    def contrast(self, ary, low, high):
        return np.array([(max(min(d, high), low) - low) / (high - low) for d in ary.reshape(-1)]).reshape(ary.shape)

    def gen_point_coords(self, res):
        cell_size = data_range * 2 / res
        cell_center = data_range - cell_size / 2
        x_list = np.linspace(-cell_center, cell_center, res)
        y_list = np.linspace(-cell_center, cell_center, res)
        point_coords = np.array([[x, y]
                                 for y_idx, y in enumerate(y_list)
                                 for x_idx, x in enumerate(x_list)])
        return point_coords


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
            for i in range(2000):
                x, y = gen.generate()
                data_set.append([list(x), y])
            json_string = json.dumps(data_set)
            with open(dataset_fielname, 'w') as f:
                f.write(json_string)
            return data_set

    def run(self):
        data_set = self.load_or_create_dataset()
        trainer = DeepTrainer(data_set, DeepLearner())
        trainer.train()


if __name__ == '__main__':
    main = Main()
    main.run()
