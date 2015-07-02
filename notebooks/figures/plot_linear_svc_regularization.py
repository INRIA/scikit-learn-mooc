import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from .plot_2d_separator import plot_2d_separator


def plot_linear_svc_regularization():
    X, y = make_blobs(centers=2, random_state=4, n_samples=30)
    # a carefully hand-designed dataset lol
    y[7] = 0
    y[27] = 0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, C in zip(axes, [1e-2, 1, 1e2]):
        ax.scatter(X[:, 0], X[:, 1], s=150, c=np.array(['red', 'blue'])[y])

        svm = SVC(kernel='linear', C=C).fit(X, y)
        plot_2d_separator(svm, X, ax=ax, eps=.5)
        ax.set_title("C = %f" % C)
