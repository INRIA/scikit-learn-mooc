"""
Simple example of overfit with splines
"""
import numpy as np
from matplotlib import pyplot as plt
import style_figs

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

# Sort the data and remove duplicates (for interpolation)
order = np.argsort(diabetes_X_train.ravel())
X_train = diabetes_X_train.ravel()[order]
y_train = diabetes_y_train[order]
# Avoid duplicates
y_train_ = list()
for this_x in np.unique(X_train):
    y_train_.append(np.mean(y_train[X_train == this_x]))
X_train = np.unique(X_train)

y_train = np.array(y_train_)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train.reshape((-1, 1)), y_train)


plt.figure(1, figsize=(.8*4, .8*3), facecolor='none')
# Plot with test data
plt.clf()
ax = plt.axes([.1, .1, .9, .9])

plt.scatter(X_train, y_train,  color='k', s=9)

plt.plot([-.08, .12], regr.predict([[-.08, ], [.12, ]]),
        linewidth=3)

plt.axis('tight')
ymin, ymax = plt.ylim()
style_figs.light_axis()
plt.ylabel('y', size=16, weight=600)
plt.xlabel('x', size=16, weight=600)

plt.savefig('ols_simple.svg', facecolor='none', edgecolor='none')

plt.scatter(diabetes_X_test, diabetes_y_test,  color='C1', s=9)
plt.ylim(ymin, ymax)
plt.xlim(-.08, .12)

plt.savefig('ols_test.svg', facecolor='none', edgecolor='none')


# Plot cubic splines
plt.clf()
ax = plt.axes([.1, .1, .9, .9])

from scipy import interpolate
f = interpolate.interp1d(X_train, y_train,
                         kind="quadratic",
                         bounds_error=False, fill_value="extrapolate")
plt.scatter(X_train, y_train,  color='k', s=9, zorder=20)
x_spline = np.linspace(-.08, .12, 600)
y_spline = f(x_spline)
plt.plot(x_spline, y_spline, linewidth=3)

plt.axis('tight')
plt.xlim(-.08, .12)
plt.ylim(ymin, ymax)

style_figs.light_axis()

plt.ylabel('y', size=16, weight=600)
plt.xlabel('x', size=16, weight=600)


plt.savefig('splines_cubic.svg', facecolor='none', edgecolor='none')


plt.scatter(diabetes_X_test, diabetes_y_test,  color='C1', s=9)
plt.savefig('splines_test.svg', facecolor='none', edgecolor='none')

plt.show()

