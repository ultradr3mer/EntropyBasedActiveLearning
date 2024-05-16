import json
import random

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier

from classifier import KnnClassifier
from deepLearner import x_resolution

data_range = 20
res = x_resolution
blob_count = 12
dataset_filename = 'dataset.json'
uint16_max = 2 ** 16 - 1

class Generator:
    def __init__(self):
        self.rnd = random.Random('dfg43')

    def generate(self):
        center_list = []
        std_list = []
        for i in range(blob_count):
            next_center = [self.rnd_float(-18, 18), self.rnd_float(-18, 18)]
            next_std = self.rnd_float(3, 5)
            if any([1 for c, std in zip(center_list, std_list)
                    if np.linalg.norm(np.array(c) - np.array(next_center)) < (next_std + std)]):
                continue
            center_list.append(next_center)
            std_list.append(next_std)

        all_x, all_y = make_blobs(n_samples=self.rnd_int(300, 600),
                                  n_features=blob_count,
                                  centers=np.array(center_list),
                                  cluster_std=np.array(std_list))
        all_x = [list(p) for p in all_x]
        y_map = {i: self.rnd_int(0, 1) for i in range(blob_count)}
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

    def load_or_create_dataset(self):
        try:
            with open(dataset_filename, 'r') as f:
                json_string = f.read()
                return json.loads(json_string)
        except FileNotFoundError:
            gen = Generator()
            data_set = []
            for i in range(2000):
                x, y = gen.generate()
                data_set.append([list(x), y])
            json_string = json.dumps(data_set)
            with open(dataset_filename, 'w') as f:
                f.write(json_string)
            return data_set

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
        labeled_i = self.means_set(all_x, random.randint(5, 20))
        remaining_x = np.array([x for i, x in enumerate(all_x) if i not in labeled_i])
        remaining_y = np.array([y for i, y in enumerate(all_y) if i not in labeled_i])
        initial_x = np.array(all_x[labeled_i])
        initial_y = np.array(all_y[labeled_i])
        point_coords = self.gen_point_coords(res)

        map_label_0 = gaussian_filter(self.histogram_class(initial_x[np.where(initial_y == 0)]), sigma=2)
        map_label_1 = gaussian_filter(self.histogram_class(initial_x[np.where(initial_y == 1)]), sigma=2)
        map_unlabeled = gaussian_filter(self.histogram_class(remaining_x), sigma=2)
        x = np.dstack((map_unlabeled, map_label_0, map_label_1))
        x = self.normalize(x)

        self.clf.fit(initial_x, initial_y)
        prior = self.clf.predict_by_point(point_coords).reshape(res, res, -1)
        prior = np.concatenate((prior,
                                np.zeros((res, res, 1))),
                               axis=2)

        def acc_for_point(p):
            acc = self.calc_acc(np.concatenate((initial_x, [p])),
                                np.concatenate((initial_y, [0])),
                                all_x,
                                all_y)
            return acc

        discovery_map = self.histogram_class([all_x[i] for i in self.means_set(all_x, 30)]).reshape(-1)

        map_unlabeled_flat = map_unlabeled.reshape(-1)
        discovery_points = np.array([i for i, coord in enumerate(point_coords) if map_unlabeled_flat[i] > 0])
        current_points = np.array([i for i in discovery_points if discovery_map[i] > 0.0])
        current_acc = np.array([acc_for_point(point_coords[i]) for i in current_points])
        remaining_points = [i for i in discovery_points if discovery_map[i] == 0.0]

        discovery_hotmap = KnnClassifier(1, 3)
        for i in range(4):
            if len(remaining_points) < 10:
                return

            discovery_hotmap.fit([point_coords[i] for i in current_points], np.ones(current_acc.shape) - current_acc)

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

        y_points = self.gen_point_coords(y_resolution)
        y = discovery_hotmap.predict(y_points).reshape(y_resolution, y_resolution)
        y = self.normalize(y)

        return x, y, prior

    def histogram_class(self, x, bins=res):
        x = np.array(x)
        H, x_edges, y_edges = np.histogram2d(x[:, 0], x[:, 1], bins=bins,
                                             range=[[-data_range, data_range], [-data_range, data_range]])
        H = np.log(H + 1)
        return H.T

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
        clf = KnnClassifier(2,3)
        clf.fit(train_x, train_y)
        return clf.prediction_acc(valid_x, valid_y)

    def save_as_image(self, x, filename):
        image = cv2.cvtColor(np.array(x * uint16_max, dtype='uint16'), cv2.CV_16U)
        cv2.imwrite(filename, image)
        pass

    def load_from_image(self, filename):
        img = cv2.imread(filename, -1)
        if img is None:
            raise FileNotFoundError(f"Image with {filename} could not be loaded.")
        return np.array(cv2.split(img), dtype=float) / uint16_max

    def gen_point_coords(self, res):
        cell_size = data_range * 2 / res
        cell_center = data_range - cell_size / 2
        x_list = np.linspace(-cell_center, cell_center, res)
        y_list = np.linspace(-cell_center, cell_center, res)
        point_coords = np.array([[x, y]
                                 for y_idx, y in enumerate(y_list)
                                 for x_idx, x in enumerate(x_list)])
        return point_coords

    def load_or_gen_train_data(self, dataset, number):
        folder = 'data2'
        x_filename = f'{folder}/{number}_x.png'
        y_filename = f'{folder}/{number}_y.png'
        p_filename = f'{folder}/{number}_p.png'
        try:
            x = self.load_from_image(x_filename)[:3]
            y = self.load_from_image(y_filename)[0]
            p = self.load_from_image(p_filename)
            return x, y, p
        except FileNotFoundError:
            x, y, p = self.gen_training_histogram(dataset)
            self.save_as_image(x, x_filename)
            self.save_as_image(y, y_filename)
            self.save_as_image(p, p_filename)
            return np.array(cv2.split(x)), y, p

if __name__ == '__main__':
    from deepTraining import DeepTrainer
    t = DeepTrainer()
    t.train()