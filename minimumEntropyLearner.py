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

    def calc_entropy(self):
        h = []
        self.classifier.fit(self.labeled_x, self.y)
        base_pred = self.classifier.predict(self.unlabeled_x)
        probs = self.estimate_probs()
        for i, x_i in enumerate(self.unlabeled_x):
            h_x = 0
            for j, label_j in enumerate(self.labels):
                prob_base = np.array([base_pred[j * self.m + i],
                                      list(self.y).count(j) / len(self.y),
                                      1 / len(self.labels)])
                prob_j = np.average(prob_base, weights=[0.0, 0.0, 1.0])
                if not prob_j:
                    continue

                h_j = calc_entropy(probs[j * self.m + i])

                h_x = h_x + prob_j * h_j
            h.append(float(h_x))
        return np.array(h)

    def build_classifier(self):
        copy = self.classifier.copy()
        copy.fit(self.labeled_x, self.y)
        return copy

    def estimate_probs(self, selected_unlabeled_x, selected_y):
        b = np.zeros((self.l * self.m, self.l * self.m))
        for i, x_i in enumerate(self.unlabeled_x):
            for j, label_j in enumerate(self.labels):
                self.classifier.fit(np.concatenate((self.labeled_x, [x_i])), np.concatenate((self.y, [label_j])))
                row = self.classifier.predict(self.unlabeled_x)
                b[j * self.m + i] = row / self.m
        return b.T * (np.matrix(np.ones((self.l * self.m, 1))) / self.l)

    def get_next_samples(self, n):
        entropy = self.calc_entropy()
        entropy_indices = list(zip(entropy, self.unlabeled_x))
        entropy_indices.sort(key=lambda x: -x[0])
        entropy_indices = entropy_indices[:n]
        return np.array([s[1] for s in entropy_indices])

    def set_labels(self, x_new, y_new):
        # x_new = []
        # y_new = []
        # for x_i, y_i in zip(x, y):
        #     if x_i in self.labeled_x:
        #         index = np.where(self.labeled_x == x_i)
        #         self.y[index[0]] = y_i
        #     x_new.append(x_i)
        #     y_new.append(y_i)

        self.labeled_x = np.concatenate((self.labeled_x, x_new), axis=0)
        self.y = np.concatenate((self.y, y_new), axis=0)
        self.unlabeled_x = np.array([single_x for single_x in self.unlabeled_x if single_x not in self.labeled_x])
        self.m = len(self.unlabeled_x)
