from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings

from classifier import SvmClassifier
from minimumEntropyLearner import MinimumEntropyLearner

warnings.filterwarnings('ignore')

# The following lines generate a random set of points in the 2D space. Please refer to make_blobs function in
# scikit-learn
x, y = make_blobs(n_samples=100, n_features=2, centers=np.array([[0, 0], [10, 18]]), cluster_std=np.array([9.0, 9.0]))
x_labeled, x_unlabeled, y_labeled, y_unlabeled = train_test_split(x, y, test_size=0.9, shuffle=True)

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

    unique_classes = np.unique(y)
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

plot_dataset(x, y)

def index_2d(a, b):
    return np.where(a == b)[0][0]

learner = MinimumEntropyLearner(SvmClassifier(), x_labeled, y_labeled, x_unlabeled, np.unique(y))
# entropy = learner.entropy_iterative()
for i in range(9):
    requested_samples = learner.get_next_samples(10)
    plot()
    learner.add_labels(requested_samples, [y[np.where(x == s)[0][0]] for s in requested_samples])

