import copy

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import warnings
from sklearn.neighbors import NearestNeighbors

from util import if_then_else

warnings.filterwarnings('ignore')

class Classifier:
    def __init__(self, l):
        self.is_fit = False
        self.l = l

    def fit(self, x, y):
        pass

    def predict(self, x):
        pass

    def prediction_prob(self, x_k, y):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def prediction_acc(self, x, y):
        m = len(x)
        pred_y = self.predict_by_point(x)
        y_probs = [[if_then_else(l == single_y, 1, 0) for l in range(self.l)] for single_y in y]
        error = np.sum(np.abs(pred_y - y_probs)) / m / self.l
        return 1 - error

    def predict_class(self, x):
        probabilities = self.predict_by_point(x)
        return np.array([np.argmax(p_i) for p_i in probabilities])

    def predict_by_point(self, x):
        if len(x) == 0:
            return np.empty((0, 2))
        return np.reshape(self.predict(x), (self.l, len(x))).T


class SvmClassifier(Classifier):
    def __init__(self, l):
        super().__init__(l)
        self.clf = svm.SVC(kernel='linear', degree=7, C=20, verbose=True)
        self.normalizer = None

    def fit(self, x, y):
        train_x = x
        train_y = y
        normalize = False
        if normalize:
            self.normalizer = StandardScaler().fit(train_x, train_y)
            data = self.normalizer.transform(train_x)
        else:
            data = train_x
        self.clf.fit(data, train_y)
        self.is_fit = True

    def predict(self, x):
        if self.normalizer is not None:
            x = self.normalizer.transform(x)
        return self.clf.predict(x)

    def prediction_prob(self, x, y):
        pred_y = self.predict(x)
        return np.array([if_then_else(label == item, 1, 0)
                         for label in range(self.l)
                         for item in pred_y])


class KnnClassifier(Classifier):
    def __init__(self, l, n):
        super().__init__(l)
        self.n = n
        self.normalizer = None
        self.x = np.empty((0, 2))
        self.y = []
        self.neighbors = None

    def fit(self, x, y):
        self.neighbors = NearestNeighbors(n_neighbors=self.n, algorithm='ball_tree').fit(x)
        self.x = x
        self.y = y
        self.is_fit = True

    def predict(self, x):
        distances, indices = self.neighbors.kneighbors(x, self.n)

        def interpolation(dist, ind, l):
            if len(ind) == 1:
                return max(1 - abs(self.y[ind] - l), 0)
            weights = max(dist) - dist
            if sum(weights) == 0:
                weights = np.ones(self.n)
            return np.average(np.array([max(1 - abs(self.y[i] - l), 0) for i in ind]), weights=weights)

        return np.array([interpolation(dist, ind, l)
                         for l in range(self.l)
                         for dist, ind in zip(distances, indices)])

    def prediction_prob(self, x, y):
        pred_y = self.predict(x)
        count = sum([1 for y in pred_y if round(y) == y])
        return count / len(x)

    def merge_points(self, x, y):
        x_new = []
        y_new = []
        for x_i, y_i in zip(x, y):
            if x_i in self.x:
                index = np.where(self.x == x_i)
                self.y[index[0]] = (self.y[index[0]] + y_i) / 2
                continue
            x_new.append(x_i)
            y_new.append(y_i)

        if len(x_new) > 0:
            self.x = np.concatenate((self.x, x_new))

        if len(y_new) > 0:
            self.y = np.concatenate((self.y, y_new))

        self.fit(self.x, self.y)

    def clear(self):
        pass

