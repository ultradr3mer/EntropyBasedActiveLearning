import copy

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import warnings
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')


def if_then_else(condition, when_true, when_false):
    if condition:
        return when_true

    return when_false


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
        pred_y = self.predict_class(x)
        return sum([1 for predicted, actual in zip(pred_y, y) if predicted == actual]) / m

    def predict_class(self, x):
        n = 3
        probabilities = self.predict_by_point(x)
        return np.array([np.where(p_i == max(p_i))[0][0] for p_i in probabilities])

    def predict_by_point(self, x):
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
        self.clf = svm.SVC(kernel='linear', degree=7, C=20, verbose=True)
        self.normalizer = None
        self.x = np.empty((0, 2))
        self.y = []
        self.neighbors = None

    def fit(self, x, y):
        self.neighbors = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(x)
        self.x = x
        self.y = y
        self.is_fit = True

    def predict(self, x):
        distances, indices = self.neighbors.kneighbors(x, self.n)
        return np.array([sum([max(1 - abs(self.y[i] - label), 0) for i in indices_k]) / self.n
                         for label in range(self.l)
                         for indices_k in indices])

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

