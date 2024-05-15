import random

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings

from classifier import SvmClassifier, KnnClassifier, Classifier
from minimumEntropyLearner import MinimumEntropyLearner
from plotFunctions import plot_dataset, plot

warnings.filterwarnings('ignore')

all_x, all_y = make_blobs(n_samples=400,
                          n_features=5,
                          centers=np.array([[-6, -1], [10, 18], [14, -6], [-12, 18], [-10, -16]]),
                          cluster_std=np.array([4.0, 6.0, 3.0, 2.5, 2.5]))
y_map = {0: 0, 1: 1, 2: 2, 3: 2, 4: 1}
all_y = [y_map[y] for y in all_y]
x_labeled, x_unlabeled, y_labeled, y_unlabeled = train_test_split(all_x, all_y, test_size=1 - 10 / 400, shuffle=True)

# plot_dataset(all_x, all_y)


def index_2d(a, b):
    return np.where(a == b)[0][0]


def init_learner():
    return MinimumEntropyLearner(KnnClassifier(len(np.unique(all_y)), 3), x_labeled, y_labeled, x_unlabeled,
                                 np.unique(all_y))


learner = init_learner()
acc_mee = []

for i in range(50):
    requested_samples = learner.get_next_samples(1)
    # if i % 10 == 0:
        # plot(requested_samples, all_y, learner)
        # plot_entropy(learner.calc_entropy(learner.unlabeled_x, learner.labeled_x, learner.y))
        # plot_decision_boundary(learner.classifier)
    learner.set_labels(requested_samples, [all_y[np.where(all_x == s)[0][0]] for s in requested_samples])
    clf = learner.build_classifier()
    acc = clf.prediction_acc(all_x, all_y)
    acc_mee.append(acc)
    print("Prediction accuracy mee", acc)
    # if i % 5 == 0:
    #     plot_entropy(learner.calc_entropy())

# plot()


acc_rnd_sum = np.array([0 for i in range(50)], dtype=float)
for run in range(10):
    learner = init_learner()
    for i in range(50):
        requested_samples = np.array([random.choice(learner.unlabeled_x)])
        # if i % 5 == 0:
            # plot(requested_samples)
        learner.set_labels(requested_samples, [all_y[np.where(all_x == s)[0][0]] for s in requested_samples])
        clf = learner.build_classifier()
        acc = clf.prediction_acc(all_x, all_y)
        acc_rnd_sum[i] = acc_rnd_sum[i] + acc / 10
        print("Prediction accuracy random", clf.prediction_acc(all_x, all_y))
        # if i % 5 == 0:
        #     plot_entropy(learner.calc_entropy())
        # acc_rnd_sum = acc_rnd_sum + acc_rnd

# plot()


def plot_acc():
    plt.plot(
        range(len(acc_mee)),
        acc_mee,
        label="mee",
        color="b",
        alpha=1.0
    )

    plt.plot(
        range(len(acc_rnd_sum)),
        acc_rnd_sum,
        label="rnd",
        color="r",
        alpha=1.0
    )

    plt.title('Data')
    plt.legend()
    plt.show()


plot_acc()
