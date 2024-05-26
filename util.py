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