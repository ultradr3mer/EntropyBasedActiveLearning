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

warnings.filterwarnings('ignore')

# The following lines generate a random set of points in the 2D space. Please refer to make_blobs function in
# scikit-learn
# all_x, all_y = make_blobs(n_samples=150, n_features=2, centers=np.array([[0, 0], [10, 18]]), cluster_std=np.array([7.0, 7.0]))
all_x, all_y = make_blobs(n_samples=400,
                          n_features=5,
                          centers=np.array([[-6, -1], [10, 18], [14, -6], [-12, 18], [-10, -16]]),
                          cluster_std=np.array([6.0, 9.0, 5.0, 4.0, 4.0]))
y_map = {0: 0, 1: 1, 2: 2, 3: 2, 4: 1}
all_y = [y_map[y] for y in all_y]
x_labeled, x_unlabeled, y_labeled, y_unlabeled = train_test_split(all_x, all_y, test_size=1-10/400, shuffle=True)


def plot_dataset(x, y):
    # This function would plot the generated points
    plt.figure()
    unique_classes = np.unique(y)
    colors = cm.magma(np.linspace(0.0, 1.0), unique_classes.size)
    rainbow = cm.get_cmap('rainbow', 4)
    for this_class in unique_classes:
        color = rainbow(this_class)
        indices = np.where(y == this_class)
        points = x[indices]
        plt.scatter(
            points[:, 0],
            points[:, 1],
            color=color,
            label="Class {}".format(this_class),
            alpha=0.5
        )
        plt.title('Data')

    plt.axis([-25.0, 35.0, -25.0, 40.0])
    plt.legend()
    plt.show()


def plot():
    plt.figure()
    rainbow = cm.get_cmap('rainbow', 4)
    plt.scatter(
        requested_samples[:, 0],
        requested_samples[:, 1],
        color="red",
        alpha=0.5,
        label="unlabeled"
    )

    unique_classes = np.unique(all_y)
    colors = cm.magma(np.linspace(0.0, 1.0), unique_classes.size)
    rainbow = cm.get_cmap('rainbow', 4)
    for this_class in unique_classes:
        color = rainbow(this_class)
        indices = np.where(learner.y == this_class)
        points = learner.labeled_x[indices]
        plt.scatter(
            points[:, 0],
            points[:, 1],
            color=color,
            label="Class {}".format(this_class),
            alpha=0.5
        )
        plt.title('Data')

    plt.axis([-25.0, 35.0, -25.0, 40.0])
    plt.title('Data')
    plt.legend()
    plt.show()


def plot_entropy(entropy):
    plt.figure()
    # unique_classes = np.unique(all_y)
    # colors = cm.magma(np.linspace(0.0, 1.0), unique_classes.size)
    # rainbow = cm.get_cmap('rainbow', 4)
    # color = rainbow(this_class)
    plt.scatter(
        learner.unlabeled_x[:, 0],
        learner.unlabeled_x[:, 1],
        c=entropy,
        label="entropy",
        alpha=1.0
    )
    plt.title('Data')

    plt.axis([-25.0, 35.0, -25.0, 40.0])
    plt.title('Data')
    plt.legend()
    plt.show()

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out
def plot_decision_boundary(clf: Classifier):
    h = 1
    region = [-25.0, 35.0, -25.0, 40.0]
    x = np.linspace(region[0], region[1], 101)
    y = np.linspace(region[2], region[3], 101)
    # full coordinate arrays
    xx, yy = np.meshgrid(x, y)
    zz = [clf.predict_class([[col, row] for col in x]) for row in y]

    # xx, yy = np.meshgrid(np.concatenate((region[:2], [h])), np.concatenate((region[2:], [h])))
    # z = [clf.predict([[row,col] for col in xx]) for row in yy]
    fig, ax = plt.subplots()
    plt.contourf(xx, yy, zz, marker='o', color='k', linestyle='none')
    # plot_contours(ax, zz, xx, yy, cmap=cm.coolwarm, alpha=1.0, normalizer=normalizer)
    # ax.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm, s=40, edgecolors='k')

    # plot support vectors
    # sx = clf.support_vectors_[:, 0]
    # sy = clf.support_vectors_[:, 1]
    # ax.scatter(sx, sy, c='yellow', edgecolors='k')
    plt.axis()

    # # plot boundary
    # if clf.kernel == 'linear':
    #     plot_bound(clf, ax, x[:, 0].min(), x[:, 0].max())

    plt.show()


plot_dataset(all_x, all_y)


def index_2d(a, b):
    return np.where(a == b)[0][0]


def init_learner():
    return MinimumEntropyLearner(KnnClassifier(len(np.unique(all_y)), 3), x_labeled, y_labeled, x_unlabeled,
                                 np.unique(all_y))

learner = init_learner()
acc_mee = []

for i in range(50):
    requested_samples = learner.get_next_samples(1)
    if i % 10 == 0:
        # plot()
        # plot_entropy(learner.calc_entropy(learner.unlabeled_x, learner.labeled_x, learner.y))
        plot_decision_boundary(learner.classifier)
    learner.set_labels(requested_samples, [all_y[np.where(all_x == s)[0][0]] for s in requested_samples])
    clf = learner.build_classifier()
    acc = clf.prediction_acc(all_x, all_y)
    acc_mee.append(acc)
    print("Prediction accuracy mee", acc)
    # if i % 5 == 0:
    #     plot_entropy(learner.calc_entropy())

# plot()

learner = init_learner()
acc_rnd = []

for i in range(50):
    requested_samples = np.array([random.choice(learner.unlabeled_x)])
    # if i % 5 == 0:
        # plot()
    learner.set_labels(requested_samples, [all_y[np.where(all_x == s)[0][0]] for s in requested_samples])
    clf = learner.build_classifier()
    acc = clf.prediction_acc(all_x, all_y)
    acc_rnd.append(acc)
    print("Prediction accuracy random", clf.prediction_acc(all_x, all_y))
    # if i % 5 == 0:
    #     plot_entropy(learner.calc_entropy())

# plot()


def plot_acc():
    plt.figure()

    plt.plot(
        range(len(acc_mee)),
        acc_mee,
        label="mee",
        color="b",
        alpha=1.0
    )

    plt.plot(
        range(len(acc_rnd)),
        acc_rnd,
        label="rnd",
        color="r",
        alpha=1.0
    )

    plt.title('Data')

    # plt.axis([-25.0, 35.0, -25.0, 40.0])
    plt.title('Data')
    plt.legend()
    plt.show()


plot_acc()
