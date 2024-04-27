import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import warnings

warnings.filterwarnings('ignore')


def if_then_else(condition, when_true, when_false):
    if condition:
        return when_true

    return when_false


class Classifier:
    def fit(self, x, y):
        pass

    def predict(self, x):
        pass

    def prediction_prob(self, x_k, label_l):
        pass

    def prediction_acc(self, other_unlabeled_x, label_l):
        pass


class SvmClassifier(Classifier):
    def __init__(self):
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

    def prediction_acc(self, x, label_l):
        pred_y = list(self.predict(x))
        return [if_then_else(y == label_l, 1, 0) for y in pred_y]

    def prediction_prob(self, x, y):
        pred_y = list(self.predict(x))
        return pred_y.count(y) / len(x)
