import json
import math
import random

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier

from classifier import KnnClassifier
from deepLearner import x_resolution, y_resolution
import plotFunctions
from pointManager import PointManager

res = x_resolution
blob_count = 12
dataset_filename = 'dataset.json'
uint16_max = 2 ** 16 - 1


class Generator:
    def __init__(self):
        self.rnd = random.Random('dfg43')
        self.clf = KnnClassifier(2, 3)
        self.classes = list(range(2))

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
        y_map[0] = 0
        y_map[1] = 1
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
        pmgr = PointManager(item)

        initial_acc = pmgr.calc_claimed(pmgr.initial_x, pmgr.initial_y, pmgr.all_x, pmgr.all_y)

        point_coords = pmgr.gen_point_coords(res)

        point_bins = [b for row in pmgr.bin(1, pmgr.remaining_x[np.where(pmgr.remaining_y == 0)])
                      for b in pmgr.bin(0, row)]
        top_y = list(enumerate(y.reshape(-1)))
        top_y.sort(key=lambda x: x[1], reverse=True)
        top_idx = np.array([i for i, y in top_y])
        top_points = np.array([p for i in top_idx for p in point_bins[i]])[:20]

        acc = sum([np.array(pmgr.claims_for_point(p)[0]) - initial_acc[0] for p in top_points]) / 20

        rnd_points = np.array(pmgr.remaining_x[np.where(pmgr.remaining_y == 0)])
        acc2 = sum([np.array(pmgr.claims_for_point(p)[0]) - initial_acc[0] for p in rnd_points]) / len(rnd_points)
        print(f"{acc} / {acc2}")

    def gen_training_histogram(self, item):
        pmgr = PointManager(item)

        x = np.dstack((pmgr.map_unlabeled, pmgr.map_label_0, pmgr.map_label_1))
        x = self.normalize(x)

        prior = np.concatenate((pmgr.calc_prior(),
                                np.zeros((res, res, 1))),
                               axis=2)

        discovery_map = pmgr.histogram_class([pmgr.all_x[i] for i in pmgr.means_set(pmgr.all_x, 80)]).reshape(-1)

        map_unlabeled_flat = pmgr.map_unlabeled.reshape(-1)
        discovery_points = np.array([i for i, coord in enumerate(pmgr.point_coords) if map_unlabeled_flat[i] > 0])
        current_points = np.array([i for i in discovery_points if discovery_map[i] > 0.0])
        current_claims = np.array([pmgr.claims_for_point(pmgr.point_coords[i]) for i in current_points]).reshape(-1, 2)
        remaining_points = [i for i in discovery_points if discovery_map[i] == 0.0]

        discovery_hotmap = KnnClassifier(1, 3)

        # for i in range(4):
        #     if len(remaining_points) < 10:
        #         return
        #
        #     discovery_hotmap.fit([pmgr.point_coords[i] for i in current_points],
        #                          [1.0 - np.average(c, weights=[1.0, 1.0]) for c in current_claims])
        #     weights = self.normalize(discovery_hotmap.predict([pmgr.point_coords[i]
        #                                                        for i in
        #                                                        remaining_points])) + 0.1
        #     if math.isnan(sum(weights)):
        #         weights = np.ones(len(weights))
        #
        #     additional_points = self.rnd.choices(remaining_points,
        #                                          weights=weights,
        #                                          k=10)
        #     current_points = np.concatenate((current_points, additional_points))
        #     new_claims = np.array([pmgr.claims_for_point(pmgr.point_coords[i]) for i in additional_points]).reshape(-1, 2)
        #     current_claims = np.concatenate((current_claims, new_claims))
        #     remaining_points = [i for i in remaining_points if i not in current_points]

        def draw_hotmap(weights):
            discovery_hotmap.fit([pmgr.point_coords[i] for i in current_points],
                                 [1.0 - np.average(c, weights=weights) for c in current_claims])
            y_points = pmgr.gen_point_coords(y_resolution)
            y = discovery_hotmap.predict(y_points).reshape(y_resolution, y_resolution)
            y = self.normalize(y)
            return y

        y_1 = draw_hotmap([1.0, 0.0])
        y_2 = draw_hotmap([0.0, 1.0])
        y_3 = draw_hotmap([len(np.where(pmgr.all_y == 0)[0]), len(np.where(pmgr.all_y == 1)[0])])
        y = np.dstack((y_1, y_2, y_3))

        return x, y, prior

    def normalize(self, x):
        x = x - np.min(x)
        max = np.max(x)
        if max != 0:
            x = x / max
        return x

    def save_as_image(self, x, filename):
        image = cv2.cvtColor(np.array(x * uint16_max, dtype='uint16'), cv2.CV_16U)
        cv2.imwrite(filename, image)
        pass

    def load_from_image(self, filename):
        img = cv2.imread(filename, -1)
        if img is None:
            raise FileNotFoundError(f"Image with {filename} could not be loaded.")
        return np.array(cv2.split(img), dtype=float) / uint16_max

    def load_or_gen_train_data(self, dataset, number):
        folder = 'data3'
        x_filename = f'{folder}/{number}_x.png'
        y_filename = f'{folder}/{number}_y.png'
        p_filename = f'{folder}/{number}_p.png'
        try:
            x = self.load_from_image(x_filename)[:3]
            y = self.load_from_image(y_filename)[:3]
            p = self.load_from_image(p_filename)[1:3]
            return x, y, p
        except FileNotFoundError:
            x, y, p = self.gen_training_histogram(dataset)
            self.save_as_image(x, x_filename)
            self.save_as_image(y, y_filename)
            self.save_as_image(p, p_filename)
            return np.array(cv2.split(x)), np.array(cv2.split(y)), np.array(cv2.split(p)[1:3])


if __name__ == '__main__':
    from deepTraining import DeepTrainer

    t = DeepTrainer()
    t.train()
