import random
import json
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

from classifier import KnnClassifier
from plotFunctions import plot_dataset, plot
from util import if_then_else

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
        self.clf = KnnClassifier(class_count, 3)

    def train(self):
        item = self.data_set[0]
        self.histogram(item)

    def histogram(self, item):
        all_x, all_y = item
        # self.clf.fit(x_labeled, y_labeled)
        # x_list = np.linspace(-20, 20, res)
        # y_list = np.linspace(-20, 20, res)
        # xx, yy = np.meshgrid(x_list, y_list)
        # request = [[x,y] for y in y_list for x in x_list]
        # predict = self.clf.predict_class(request)
        # zz = predict.reshape((res, res))
        all_x = np.array(all_x)
        all_y = np.array(all_y)
        classes = np.unique(all_y)
        labeled_i = self.initial_set(all_x, 10)
        remaining_x = np.array([x for i, x in enumerate(all_x) if i not in labeled_i])
        remaining_y = np.array([y for i, y in enumerate(all_y) if i not in labeled_i])
        initial_x = np.array(all_x[labeled_i])
        initial_y = np.array(all_y[labeled_i])
        map_label_0 = self.histogram_class(initial_x[np.where(initial_y == 0)])
        map_label_1 = self.histogram_class(initial_x[np.where(initial_y == 1)])
        map_unlabeled = self.histogram_class(remaining_x)
        # H, x_edges, y_edges = np.histogram2d(all_x[:,0], all_x[:,1], bins=res, range=[[-range, range], [-range, range]])
        # H = np.log(H + 1)
        x = np.array(list(zip(map_unlabeled.reshape(-1), map_label_0.reshape(-1), map_label_1.reshape(-1))))
        x = self.normalize_map(x)

        initial_acc = self.calc_acc(initial_x, initial_y, all_x, all_y)
        # y_acc = np.array([self.calc_acc(np.concatenate((initial_x, xy[0].reshape(-1,2)), axis=0),
        #                                 np.concatenate((initial_y, [remaining_y[0]])),
        #                                 all_x,
        #                                 all_y)
        #                   for xy in zip(remaining_x, remaining_y)])
        cell_size = data_range * 2 / res
        cell_center = data_range - cell_size / 2
        x_list = np.linspace(-cell_center, cell_center, res)
        y_list = np.linspace(-cell_center, cell_center, res)
        acc_map = np.array([[if_then_else(map_unlabeled[y_idx][x_idx] > 0.0, max(max(
            [self.calc_acc(np.concatenate((initial_x, np.array([x, y]).reshape(1, 2))),
                           np.concatenate((initial_y, [c])),
                           all_x,
                           all_y) for c in classes]) - initial_acc, 0), 0)
                             for x_idx, x in enumerate(x_list)] for y_idx, y in enumerate(y_list)]).reshape(-1)
        acc_map = normalize(acc_map.reshape(-1, 1), norm='max', axis=0).reshape(res, res)
        # acc_map = self.max_y_map(points)

        plt.imshow(acc_map, origin='lower', extent=[-data_range, data_range, -data_range, data_range], cmap='viridis')
        plt.colorbar(label='Counts')
        plt.xlabel('X values')
        plt.ylabel('Y values')
        plt.title('2D Histogram')
        plt.show()
        plot(initial_x, remaining_x, remaining_y)

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

    def normalize_map(self, x):
        return normalize(x.reshape(-1, 1), norm='max', axis=0).reshape(res, -1) * 2 - 1

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

    def initial_set(self, x, n):
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(x)
        centroids = kmeans.cluster_centers_
        indices = np.array([np.argmin([np.linalg.norm(x - c) for i, x in enumerate(x)]) for c in centroids])
        return indices

    def calc_acc(self, train_x, train_y, valid_x, valid_y):
        self.clf.fit(train_x, train_y)
        return self.clf.prediction_acc(valid_x, valid_y)


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
