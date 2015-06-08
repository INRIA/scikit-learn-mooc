"""
Small helpers for code that is not shown in the notebooks
"""

from sklearn import neighbors, datasets, linear_model
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Create color maps for 3-class classification problem, as with iris
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


def plot_iris_knn():
    iris = datasets.load_iris()
    # we only take the first two features. We could
    # avoid this ugly slicing by using a two-dim dataset
    X = iris.data[:, :2]

    y = iris.target

    knn = neighbors.KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')
    plt.axis('tight')


def plot_polynomial_regression():
    rng = np.random.RandomState(0)
    x = 2 * rng.rand(100) - 1

    f = lambda t: 1.2 * t**2 + .1 * t**3 - .4 * t ** 5 - .5 * t ** 9
    y = f(x) + .4 * rng.normal(size=100)

    x_test = np.linspace(-1, 1, 100)

    plt.figure()
    plt.scatter(x, y, s=4)

    X = np.array([x**i for i in range(5)]).T
    X_test = np.array([x_test**i for i in range(5)]).T
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    plt.plot(x_test, regr.predict(X_test), label='4th order')

    X = np.array([x**i for i in range(10)]).T
    X_test = np.array([x_test**i for i in range(10)]).T
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    plt.plot(x_test, regr.predict(X_test), label='9th order')

    plt.legend(loc='best')
    plt.axis('tight')
    plt.title('Fitting a 4th and a 9th order polynomial')

    plt.figure()
    plt.scatter(x, y, s=4)
    plt.plot(x_test, f(x_test), label="truth")
    plt.axis('tight')
    plt.title('Ground truth (9th order polynomial)')
