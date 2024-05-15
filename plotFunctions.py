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


def plot_dataset(x, y):
    # This function would plot the generated points
    plt.figure()
    unique_classes = np.unique(y)
    rainbow = plt.colormaps['rainbow'].resampled(max(unique_classes))
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

    plt.axis([-22.0, 22.0, -22.0, 22.0])
    plt.legend()
    plt.show()


def plot(requested_samples, labeled_x, y):
    plt.figure()
    rainbow = cm.get_cmap('rainbow', 4)

    if requested_samples is not None:
        plt.scatter(
            requested_samples[:, 0],
            requested_samples[:, 1],
            color="red",
            alpha=0.5,
            label="unlabeled"
        )

    unique_classes = np.unique(y)
    colors = cm.magma(np.linspace(0.0, 1.0), unique_classes.size)
    rainbow = cm.get_cmap('rainbow', 4)
    for this_class in unique_classes:
        color = rainbow(this_class)
        indices = np.where(y == this_class)
        points = labeled_x[indices]
        plt.scatter(
            points[:, 0],
            points[:, 1],
            color=color,
            label="Class {}".format(this_class),
            alpha=0.5
        )
        plt.title('Data')

    plt.axis([-20.0, 20.0, -20.0, 20.0])
    plt.title('Data')
    plt.legend()
    plt.show()


def plot_entropy(entropy):
    plt.figure()
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
    fig, ax = plt.subplots()
    plt.contourf(xx, yy, zz, marker='o', color='k', linestyle='none')
    plt.axis()
    plt.show()


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


def plot_img(img):
    plt.imshow(img, origin='lower', cmap='viridis')
    plt.colorbar(label='Counts')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title('2D Histogram')
    plt.show()
