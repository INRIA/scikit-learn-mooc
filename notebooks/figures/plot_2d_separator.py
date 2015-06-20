import numpy as np
import matplotlib.pyplot as plt


def plot_2d_separator(classifier, X, fill=False):
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx = np.linspace(x_min, x_max, 10)
    yy = np.linspace(y_min, y_max, 10)

    X1, X2 = np.meshgrid(xx, yy)
    X_grid = np.c_[X1.ravel(), X2.ravel()]
    try:
        decision_values = classifier.decision_function(X_grid)
    except AttributeError:
        # no decision_function
        decision_values = classifier.predict_proba(X_grid)[:, 0]

    ax = plt.axes()
    if fill:
        ax.contourf(X1, X2, decision_values.reshape(X1.shape))
    else:
        ax.contour(X1, X2, decision_values.reshape(X1.shape), levels=[0],
                   colors="black")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from sklearn.linear_model import LogisticRegression
    X, y = make_blobs(centers=2, random_state=42)
    clf = LogisticRegression().fit(X, y)
    plot_2d_separator(clf, X, fill=True)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()
