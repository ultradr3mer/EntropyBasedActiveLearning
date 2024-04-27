import random

from classifier import Classifier, KnnClassifier
import numpy as np
from sklearn import preprocessing as pre


def calc_entropy(p):
    if (p == 0 or p == 1):
        return 0
    return -p * np.log2(p)


def if_then_else(condition, when_true, when_false):
    if condition:
        return when_true

    return when_false


class MinimumEntropyLearner:
    def __init__(self, classifier: Classifier, labeled_x, y, unlabeled_x, labels):
        self.labels = labels
        self.unlabeled_x = unlabeled_x
        self.y = y
        self.labeled_x = labeled_x
        self.classifier = classifier
        self.l = len(labels)
        self.entropy_heatmap: KnnClassifier = KnnClassifier(1, 5)

    def calc_entropy(self, selected_unlabeled_x, labeled_x, y):
        m = len(selected_unlabeled_x)
        h = []
        self.classifier.fit(labeled_x, y)
        base_pred = self.classifier.predict(selected_unlabeled_x)
        probs = self.estimate_probs(selected_unlabeled_x, labeled_x, y)
        for i, x_i in enumerate(selected_unlabeled_x):
            h_x = 0
            for j, label_j in enumerate(self.labels):
                prob_base = np.array([base_pred[j * m + i],
                                      list(y).count(j) / len(y),
                                      1 / len(self.labels)])
                prob_j = np.average(prob_base, weights=[0.0, 0.0, 1.0])
                if not prob_j:
                    continue

                h_j = calc_entropy(probs[j * m + i])

                h_x = h_x + prob_j * h_j
            h.append(float(h_x))
        return np.array(h)

    def build_classifier(self):
        copy = self.classifier.copy()
        copy.fit(self.labeled_x, self.y)
        return copy

    def estimate_probs(self, selected_unlabeled_x, labeled_x, y):
        m = len(selected_unlabeled_x)
        b = np.zeros((self.l * m, self.l * m))
        for i, x_i in enumerate(selected_unlabeled_x):
            for j, label_j in enumerate(self.labels):
                self.classifier.fit(np.concatenate((labeled_x, [x_i])), np.concatenate((y, [label_j])))
                row = self.classifier.predict(selected_unlabeled_x)
                b[j * m + i] = row / m
        return b.T * (np.matrix(np.ones((self.l * m, 1))) / self.l)

    def get_next_samples(self, n, discover=20):
        picked_points = self.pick_unlabeled(discover)
        entropy = self.calc_entropy(picked_points, self.labeled_x, self.y)
        entropy_indices = list(zip(entropy, picked_points))
        entropy_indices.sort(key=lambda x: -x[0])
        entropy_indices = entropy_indices[:n]
        self.entropy_heatmap.fit(picked_points, entropy)
        return np.array([s[1] for s in entropy_indices])

    def set_labels(self, x, y):
        self.labeled_x = np.concatenate((self.labeled_x, x), axis=0)
        self.y = np.concatenate((self.y, y), axis=0)
        self.unlabeled_x = np.array([single_x for single_x in self.unlabeled_x if single_x not in self.labeled_x])
        m = len(self.unlabeled_x)

    def pick_unlabeled(self, n):
        points = np.empty((0, 2))
        if self.entropy_heatmap.is_fit:
            entropy = self.entropy_heatmap.predict_by_point(self.unlabeled_x) * -1
            entropy = pre.MinMaxScaler(feature_range=(0.2, 1.0)).fit_transform(entropy)
            return random.choices(self.unlabeled_x, weights=entropy, k=n)
        else:
            return random.sample(list(self.unlabeled_x), n)
