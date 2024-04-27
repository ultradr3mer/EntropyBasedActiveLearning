from classifier import Classifier
import numpy as np


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
        self.m = len(unlabeled_x)

    def entropy_iterative(self):
        h = []
        for i, x_i in enumerate(self.unlabeled_x):
            h_x = 0
            for j, label_j in enumerate(self.labels):
                self.classifier.fit(self.labeled_x, self.y)
                prob_j = self.classifier.prediction_prob([x_i], j)
                if not prob_j:
                    continue

                # prob_j = 1 # list(self.y).count(label_j) / len(self.y)
                self.classifier.fit(self.labeled_x + [x_i], self.y + [label_j])
                other_unlabeled_x = np.concatenate((self.unlabeled_x[:i, :], self.unlabeled_x[i + 1:, :]), axis=0)
                h_j = sum([calc_entropy(self.classifier.prediction_prob(other_unlabeled_x, label_l))
                           for label_l in self.labels])
                h_x = h_x + prob_j * h_j
            h.append(h_x)
        return h

    def entropy_by_mat(self):
        h = []
        probs = self.estimate_probs()
        self.classifier.fit(self.labeled_x, self.y)
        base_pred = self.classifier.predict(self.unlabeled_x)
        for i, x_i in enumerate(self.unlabeled_x):
            h_x = 0
            for j, label_j in enumerate(self.labels):
                prob_base = np.array([if_then_else(base_pred[i] == label_j, 1, 0),
                                      list(self.y).count(j) / len(self.y),
                                      1 / len(self.labels)])
                prob_j = np.average(prob_base, weights=[0.7, 0.2, 0.1])
                if not prob_j:
                    continue

                h_j = sum([calc_entropy(probs[l * self.m + i])
                           for l, label_l in enumerate(self.labels)])
                h_x = h_x + prob_j * h_j
            h.append(h_x)
        return h

    def estimate_probs(self):
        b = np.zeros((self.l * self.m, self.l * self.m))
        for i, x_i in enumerate(self.unlabeled_x):
            for j, label_j in enumerate(self.labels):
                self.classifier.fit(self.labeled_x + [x_i], self.y + [label_j])
                pred = self.classifier.predict(self.unlabeled_x)
                row = (np.array([if_then_else(pred_item == label_l, 1, 0)
                                 for pred_item in pred
                                 for label_l in self.labels])
                       )
                b[j * self.m + i] = row
        return b * np.matrix(np.ones((self.l * self.m, 1))) / self.l / self.m

    def get_next_samples(self, n):
        entropy = self.entropy_by_mat()
        entropy_indices = list(zip(entropy, self.unlabeled_x))
        entropy_indices.sort(key=lambda x: -x[0])
        entropy_indices = entropy_indices[:n]
        return np.array([s[1] for s in entropy_indices])

    def add_labels(self, x_new, y_new):
        self.labeled_x = np.concatenate((self.labeled_x, x_new), axis=0)
        self.y = np.concatenate((self.y, y_new), axis=0)
        self.unlabeled_x = np.array([single_x for single_x in self.unlabeled_x if single_x not in self.labeled_x])
        self.m = len(self.unlabeled_x)
