"""
An introduction to OLS
"""

import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets, linear_model

# Load the diabetes dataset
diabetes = datasets.load_diabetes()


# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis]
diabetes_X_temp = diabetes_X[:, :, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X_temp[:-200:3]
diabetes_X_test = diabetes_X_temp[-200:].T

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-200:3]
diabetes_y_test = diabetes.target[-200:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

pl.figure(1, figsize=(4, 3), facecolor='k')
# Plot with test data
pl.clf()
ax = pl.axes([.1, .1, .9, .9], axisbg='k')

pl.scatter(diabetes_X_train, diabetes_y_train,  color='w', s=4)

pl.plot([-.08, .12], regr.predict([[-.08, ], [.12, ]]), color='c',
        linewidth=3)

pl.axis('tight')
ymin, ymax = pl.ylim()
pl.scatter(diabetes_X_test, diabetes_y_test,  color='m', s=6)
pl.ylim(ymin, ymax)
pl.xlim(-.08, .12)

pl.xticks(())
pl.yticks(())
ax.spines['bottom'].set_color('w')
ax.spines['left'].set_color('w')
pl.ylabel('y', color='w', size=16, weight=600)
pl.xlabel('x', color='w', size=16, weight=600)

pl.savefig('ols_test.pdf', facecolor='k', edgecolor='k')


pl.clf()
ax = pl.axes([.1, .1, .9, .9], axisbg='k')

pl.scatter(diabetes_X_train, diabetes_y_train,  color='w', s=4)
pl.plot([-.08, .12], regr.predict([[-.08, ], [.12, ]]), color='c',
        linewidth=3)

pl.axis('tight')
pl.xlim(-.08, .12)

pl.xticks(())
pl.yticks(())
pl.savefig('ols_simple.pdf', facecolor='k', edgecolor='k')

ax.spines['bottom'].set_color('w')
ax.spines['left'].set_color('w')
pl.ylabel('y', color='w', size=16, weight=600)
pl.xlabel('x', color='w', size=16, weight=600)

pl.savefig('ols.pdf', facecolor='k', edgecolor='k')

# Plot lines symbolizing the fit:
for x, y in zip(diabetes_X_train, diabetes_y_train):
    pl.plot([x, x], [y, regr.predict([[x, ]])], color='.8', linewidth=1)

pl.savefig('ols_error.pdf', facecolor='k', edgecolor='k')
ymin, ymax = pl.ylim()

# Plot smoothing splines
pl.clf()
ax = pl.axes([.1, .1, .9, .9], axisbg='k')

pl.scatter(diabetes_X_train, diabetes_y_train,  color='w', s=7,
           zorder=20)
from scipy import interpolate
order = np.argsort(diabetes_X_train.ravel())
X_clean = diabetes_X_train.ravel()[order]
y_clean = diabetes_y_train[order]
# Avoid duplicates
y_clean_ = list()
for this_x in np.unique(X_clean):
    y_clean_.append(np.mean(y_clean[X_clean == this_x]))
f = interpolate.UnivariateSpline(np.unique(X_clean), y_clean_, s=0)
x_spline = np.linspace(-.08, .12)
y_spline = f(x_spline)
pl.plot(x_spline, y_spline, color='c', linewidth=3)

pl.axis('tight')
pl.xlim(-.08, .12)
pl.ylim(ymin, ymax)

pl.xticks(())
pl.yticks(())

ax.spines['bottom'].set_color('w')
ax.spines['left'].set_color('w')
pl.ylabel('y', color='w', size=16, weight=600)
pl.xlabel('x', color='w', size=16, weight=600)

pl.savefig('splines.pdf', facecolor='k', edgecolor='k')

for s, c in zip((12, 20), ('y', 'm', 'g')):
    f = interpolate.UnivariateSpline(np.unique(X_clean), y_clean_,
                    s=10000.*s)
    y_spline = f(x_spline)
    pl.plot(x_spline, y_spline, color=c, linewidth=2.5)

pl.savefig('splines_smoothed.pdf', facecolor='k', edgecolor='k')

# Plot with test data
pl.clf()
ax = pl.axes([.1, .1, .9, .9], axisbg='k')

pl.scatter(diabetes_X_train, diabetes_y_train,  color='w', s=4)
f = interpolate.UnivariateSpline(np.unique(X_clean), y_clean_, s=0)
x_spline = np.linspace(-.08, .12)
y_spline = f(x_spline)
pl.plot(x_spline, y_spline, color='c', linewidth=3)

pl.axis('tight')
pl.xlim(-.08, .12)
pl.ylim(ymin, ymax)

pl.xticks(())
pl.yticks(())

ax.spines['bottom'].set_color('w')
ax.spines['left'].set_color('w')
pl.ylabel('y', color='w', size=16, weight=600)
pl.xlabel('x', color='w', size=16, weight=600)

pl.scatter(diabetes_X_test, diabetes_y_test,  color='m', s=6)
pl.savefig('splines_test.pdf', facecolor='k', edgecolor='k')

###############################################################################
# Now the 3D figure
diabetes_X = diabetes.data[:]
diabetes_X_temp = diabetes_X[:, :2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X_temp[:-200:3]
diabetes_X_test = diabetes_X_temp[-200:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-200:3]
diabetes_y_test = diabetes.target[-200:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)


fig = pl.figure(2, figsize=(4, 3), facecolor='k', edgecolor='w')

for name, x1, x2, y in [
                     ('3d', diabetes_X_train[:, 0], diabetes_X_train[:, 1],
                      diabetes_y_train),
                     ('3d_empty', [0.025, 0.05],
                                  [-0.045, -0.015],
                                  [246., 162.],
                     )
                    ]:
    pl.clf()
    ax = Axes3D(fig, elev=43.5, azim=-110)

    ax.scatter(x1, x2, y, c='w', marker='+', s=50, linewidth=2)
    ax.plot_surface(np.array([[-.1, -.1], [.15, .15]]),
                    np.array([[-.1, .15], [-.1, .15]]),
                    regr.predict(np.array([[-.1, -.1, .15, .15],
                                            [-.1, .15, -.1, .15]]).T
                                ).reshape((2, 2)),
                    alpha=.5)

    ax.set_axis_bgcolor('k')
    ax.set_xlabel('X_1', color='w', weight=600)
    ax.set_ylabel('X_2', color='w', weight=600)
    ax.set_zlabel('y', color='w', weight=600)
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    pl.savefig('ols_%s.pdf' % name, facecolor='k', edgecolor='k')


pl.show()

