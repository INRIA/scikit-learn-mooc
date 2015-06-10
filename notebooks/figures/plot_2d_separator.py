import numpy as np
import matplotlib.pyplot as plt


def plot_2d_separator(classifier, X):
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    xx = np.linspace(x_min, x_max, 10)
    yy = np.linspace(y_min, y_max, 10)

    X1, X2 = np.meshgrid(xx, yy)
    Z = np.empty(X1.shape)
    for (i, j), val in np.ndenumerate(X1):
        x1 = val
        x2 = X2[i, j]
        try:
            p = classifier.decision_function([x1, x2])
        except AttributeError:
            # no decision_function
            p = classifier.predict_proba([x1, x2])[:, 0]
        Z[i, j] = p[0]
    levels = [0.0]
    linestyles = ['dashed', 'solid', 'dashed']
    colors = 'k'

    ax = plt.axes()
    ax.contour(X1, X2, Z, levels, colors=colors, linestyles=linestyles)


if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from sklearn.linear_model import LogisticRegression
    X, y = make_blobs(centers=2, random_state=42)
    clf = LogisticRegression().fit(X, y)
    plot_2d_separator(clf, X)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()
