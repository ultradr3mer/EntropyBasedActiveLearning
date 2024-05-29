import numpy as np


def if_then_else(condition, when_true, when_false):
    if condition:
        return when_true

    return when_false


def normalize(x):
    x = x - np.min(x)
    max = np.max(x)
    if max != 0:
        x = x / max
    return x


def prob_from_class(count, class_nr):
    if class_nr is not np.array:
        class_nr = np.array(class_nr)
    if len(class_nr) == 0:
        return np.empty(0, count)
    shape = class_nr.shape
    class_nr = class_nr.reshape(-1)
    return np.array([max(1 - abs(c - l), 0)
                     for c in class_nr
                     for l in range(count)],
                    dtype=np.float32).reshape(*shape, -1)
