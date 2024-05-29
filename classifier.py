import copy

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import warnings
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

from util import if_then_else, prob_from_class

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
        y_probs = prob_from_class(self.l, y)
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
    def __init__(self, l, k):
        super().__init__(l)
        self.k = k
        self.normalizer = None
        self.x = np.empty((0, 2))
        self.y = []
        self.prob_smooth = []
        self.neighbors = None

    def fit(self, x, y):
        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)
        self.neighbors = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree').fit(self.x)
        # self.prob_smooth = np.average(prob_from_class(self.l, self.y[self.neighbors.kneighbors(x, self.k)[1]]), axis=1)
        self.is_fit = True

    def predict(self, x):
        distances, indices = self.neighbors.kneighbors(x, self.k)

        def interpolation(dist, probs):
            count = len(probs)
            # if count != len(dist):
            #     raise ValueError('asdasdas')
            if count == 1:
                return probs[0]

            # touching_points = np.where(dist == 0)
            # if len(np.where(dist == 0)[0]) > 0:
            #     return np.average(probs[touching_points], axis=0)

            weights = max(dist) - dist
            if sum(weights) == 0:
                weights = np.ones(count)
            # weights = np.log([1 + np.prod([d for i, d in enumerate(dist) if i != dist_i]) for dist_i in range(count)])
            return np.average(probs, weights=weights, axis=0)

        return np.array([interpolation(dist, probs)
                         for dist, probs in zip(distances, prob_from_class(self.l, self.y[indices]))])

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
