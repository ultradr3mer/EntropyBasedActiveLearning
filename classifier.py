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
        self.l = l

    def fit(self, x, y):
        pass

    def predict(self, x):
        pass

    def prediction_prob(self, x_k, y):
        pass

    # def prediction_acc(self, other_unlabeled_x, y):
    #     pass


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

    def predict(self, x):
        if self.normalizer is not None:
            x = self.normalizer.transform(x)
        return self.clf.predict(x)

    # def prediction_acc(self, x, y):
    #     pred_y = list(self.predict(x))
    #     return [if_then_else(y == y, 1, 0) for y in pred_y]

    def prediction_prob(self, x, y):
        pred_y = self.predict(x)
        return np.array([if_then_else(label == item, 1, 0)
                         for label in range(self.l)
                         for item in pred_y])


class KnnClassifier(Classifier):
    def __init__(self, l):
        super().__init__(l)
        self.clf = svm.SVC(kernel='linear', degree=7, C=20, verbose=True)
        self.normalizer = None
        self.x = None
        self.y = None
        self.neighbors = None

    def fit(self, x, y):
        self.neighbors = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(x)
        self.x = x
        self.y = y

    def predict(self, x):
        n = 3
        distances, indices = self.neighbors.kneighbors(x, n)
        return np.array([sum([1 for i in indices_k if self.y[i] == label]) / n
                         for label in range(self.l)
                         for indices_k in indices])

    # def prediction_acc(self, x, y):
    #     pred_y = self.predict(x)
    #     return [if_then_else(round(y) == y, 1, 0) for y in pred_y]

    def prediction_prob(self, x, y):
        pred_y = self.predict(x)
        count = sum([1 for y in pred_y if round(y) == y])
        return count / len(x)
