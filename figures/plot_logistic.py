"""
An introduction to OLS
"""

import numpy as np
import pylab as pl

from sklearn import datasets, linear_model, svm

def decorate(xlabel='x', ylabel='y'):
    ax = pl.gca()
    pl.axis('tight')
    ax.spines['bottom'].set_color('w')
    ax.spines['left'].set_color('w')
    pl.xticks(())
    pl.yticks(())
    pl.xlabel(xlabel, color='w', size=16, weight=600)
    pl.ylabel(ylabel, color='w', size=16, weight=600)


xmin, xmax = -5, 5
n_samples = 100
np.random.seed(0)
X = np.random.normal(size=n_samples)
y = (X > 0).astype(np.float)
X[X > 0] *= 4
X += .3 * np.random.normal(size=n_samples)

X_train = X[:, np.newaxis]
y_train = y

# Create linear regression object
regr = linear_model.LogisticRegression(C=1e5)

# Train the model using the training sets
regr.fit(X_train, y_train)

# plot the line, the points, and the nearest vectors to the plane
pl.figure(1, figsize=(4, 3), facecolor='k')
pl.clf()
ax = pl.axes([.1, .1, .9, .9], axisbg='k')

pl.scatter(X_train, y_train,  color='w', s=25, zorder=20)

decorate()
pl.xlim(-3, 10)
pl.savefig('categorical.pdf', facecolor='k', edgecolor='k')

def model(x):
    return 1 / (1 + np.exp(-x))

X_test = np.linspace(-4, 10, 300)
loss = model(X_test * regr.coef_ + regr.intercept_).ravel()

pl.plot(X_test, loss, color='c', linewidth=3)

decorate()
pl.xlim(-3, 10)
pl.savefig('logistic.pdf', facecolor='k', edgecolor='k')

pl.scatter(X.ravel(), y, c=y, zorder=25, s=40, edgecolor='w', cmap=pl.cm.Paired)

pl.xlim(-3, 10)
pl.savefig('logistic_color.pdf', facecolor='k', edgecolor='k')

# A 2D version
pl.figure(2, figsize=(3, 3), facecolor='k')

# A variety of different datasets
data = {}

# Very sharp and somewhat stupid transition
X = np.random.normal(size=(n_samples, 2))
y = (X[:, 0] + 1.4 * X[:, 1] > 0).astype(np.float)
X[y > 0] *= 4
X += .3 * np.random.normal(size=X.shape)
data['sharp'] = (X, y)

# 2 Gaussian blobs
X = np.r_[
          .75 * np.random.normal(size=(n_samples / 2, 2)),
          .75 * np.random.normal(size=(n_samples / 2, 2)) + np.r_[1, 2.8]
         ]

X[:, 1] *= .5

y = np.arange(n_samples) > n_samples / 2
data['gaussian'] = (X, y)

X = np.r_[
          .75 * np.random.normal(size=(n_samples / 2, 2)),
          .75 * np.random.normal(size=(n_samples / 2, 2)) + np.r_[1, 2.8]
         ]

X[:, 1] *= .5
X[n_samples / 2:, 0] *= 2
data['gaussian_ani'] = (X, y)

iris = datasets.load_iris()
X = iris.data[np.logical_or(iris.target == 2, iris.target == 1)]
y = iris.target[np.logical_or(iris.target == 2, iris.target == 1)]

X = X[:, :2]

data['iris'] = (X, y)

for i, (name, (X, y)) in enumerate(sorted(data.items())):
    pl.clf()
    ax = pl.axes([.1, .1, .9, .9], axisbg='k')
    pl.scatter(X[:, 0], X[:, 1], c=y, zorder=25, s=40, edgecolor='w',
               cmap=pl.cm.Paired)

    decorate(xlabel='X_1', ylabel='X_2')
    pl.savefig('classification_2D_%i.pdf' % i, facecolor='k', edgecolor='k')

###############################################################################
# The SVM

_, (X, y) = sorted(data.items())[1]

pl.clf()
ax = pl.axes([.1, .1, .9, .9], axisbg='k')
pl.scatter(X[:, 0], X[:, 1], c=y, zorder=25, s=40, edgecolor='w',
            cmap=pl.cm.Paired)

decorate(xlabel='X_1', ylabel='X_2')

# fit the model
clf = svm.SVC(kernel='linear')
clf.fit(X, y)

# get the separating hyperplane
w =  clf.coef_[0]
a = -w[0]/w[1]
xx = np.linspace(-5, 5)
yy = a*xx - (clf.intercept_[0])/w[1]

# plot the parallels to the separating hyperplane that pass through the
# support vectors
b = clf.support_vectors_[0]
yy_down = a*xx + (b[1] - a*b[0])
b = clf.support_vectors_[-1]
yy_up = a*xx + (b[1] - a*b[0])

pl.plot(xx, yy, 'w-')
pl.plot(xx, yy_down, 'w--')
pl.plot(xx, yy_up, 'w--')


pl.savefig('classification_svm.pdf', facecolor='k', edgecolor='k')

pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
           s=110, facecolors='none', edgecolor='m', zorder=100)

pl.savefig('classification_svm_sv.pdf', facecolor='k', edgecolor='k')

pl.show()
