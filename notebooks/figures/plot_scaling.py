import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.model_selection import train_test_split
from .plot_helpers import cm2


def plot_scaling():
    X, y = make_blobs(n_samples=50, centers=2, random_state=4, cluster_std=1)
    X += 3

    plt.figure(figsize=(15, 8))
    main_ax = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=2)

    main_ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm2, s=60)
    maxx = np.abs(X[:, 0]).max()
    maxy = np.abs(X[:, 1]).max()

    main_ax.set_xlim(-maxx + 1, maxx + 1)
    main_ax.set_ylim(-maxy + 1, maxy + 1)
    main_ax.set_title("Original Data")
    other_axes = [plt.subplot2grid((2, 4), (i, j)) for j in range(2, 4) for i in range(2)]

    for ax, scaler in zip(other_axes, [StandardScaler(), RobustScaler(),
                                       MinMaxScaler(), Normalizer(norm='l2')]):
        X_ = scaler.fit_transform(X)
        ax.scatter(X_[:, 0], X_[:, 1], c=y, cmap=cm2, s=60)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_title(type(scaler).__name__)

    other_axes.append(main_ax)

    for ax in other_axes:
        ax.spines['left'].set_position('center')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('center')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')


def plot_relative_scaling():
    # make synthetic data
    X, _ = make_blobs(n_samples=50, centers=5, random_state=4, cluster_std=2)
    # split it into training and test set
    X_train, X_test = train_test_split(X, random_state=5, test_size=.1)
    # plot the training and test set
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].scatter(X_train[:, 0], X_train[:, 1],
                    c='b', label="training set", s=60)
    axes[0].scatter(X_test[:, 0], X_test[:, 1], marker='^',
                    c='r', label="test set", s=60)
    axes[0].legend(loc='upper left')
    axes[0].set_title("original data")

    # scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # visualize the properly scaled data
    axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1],
                    c='b', label="training set", s=60)
    axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], marker='^',
                    c='r', label="test set", s=60)
    axes[1].set_title("scaled data")

    # rescale the test set separately, so that test set min is 0 and test set max is 1
    # DO NOT DO THIS! For illustration purposes only
    test_scaler = MinMaxScaler()
    test_scaler.fit(X_test)
    X_test_scaled_badly = test_scaler.transform(X_test)

    # visualize wrongly scaled data
    axes[2].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1],
                    c='b', label="training set", s=60)
    axes[2].scatter(X_test_scaled_badly[:, 0], X_test_scaled_badly[:, 1], marker='^',
                    c='r', label="test set", s=60)
    axes[2].set_title("improperly scaled data")
