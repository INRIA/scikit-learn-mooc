import numpy as np
from matplotlib import pyplot as plt

# Set up figures look and feel
import style_figs

# %%
# Our data-generating process

def f(t):
    return 1.2 * t ** 2 + .1 * t ** 3 - .4 * t ** 5 - .5 * t ** 9

N_SAMPLES = 50

rng = np.random.RandomState(0)
x = 2 * rng.rand(N_SAMPLES) - 1

y = f(x) + .4 * rng.normal(size=N_SAMPLES)

plt.figure()
plt.scatter(x, y, s=20, color='k')

style_figs.no_axis()
plt.subplots_adjust(top=.96)
plt.xlim(-1.1, 1.1)
plt.ylim(-.74, 2.1)
plt.savefig('polynomial_overfit_0.svg', facecolor='none', edgecolor='none')

# %%
# Our model (polynomial regression)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# %%
# Fit model with various complexity in the polynomial degree

plt.figure()
plt.scatter(x, y, s=20, color='k')

t = np.linspace(-1, 1, 100)

for d in (1, 2, 5, 9):
    model = make_pipeline(PolynomialFeatures(degree=d), LinearRegression())
    model.fit(x.reshape(-1, 1), y)
    plt.plot(t, model.predict(t.reshape(-1, 1)), label='Degree %d' % d,
             linewidth=4)

    style_figs.no_axis()
    plt.legend(loc='upper center', borderaxespad=0, borderpad=0)
    plt.subplots_adjust(top=.96)
    plt.ylim(-.74, 2.1)

    plt.savefig('polynomial_overfit_%d.svg' % d, facecolor='none',
                edgecolor='none')

plt.plot(t, f(t), 'k--', label='Truth')

style_figs.no_axis()
plt.legend(loc='upper center', borderaxespad=0, borderpad=0)
plt.ylim(-.74, 2.1)
plt.subplots_adjust(top=.96)

plt.savefig('polynomial_overfit.svg', facecolor='none', edgecolor='none')

# %%
# A figure with the true model and the estimated one

plt.figure(figsize=[.5 * 6.4, .5 * 4.9])
plt.scatter(x, y, s=20, color='k')
plt.plot(t, model.predict(t.reshape(-1, 1)), color='C3',
         label='$\hat{f}$')

plt.plot(t, f(t), 'k--', label='$f^{\star}$')
style_figs.no_axis()
plt.ylim(-1.25, 2.5)
plt.legend(loc='upper center', borderaxespad=0, borderpad=0,
           labelspacing=.4, fontsize=26)
plt.subplots_adjust(top=1)

plt.savefig('polynomial_overfit_simple.svg', facecolor='none', edgecolor='none')

# %%
# A figure with the true model and the estimated one

plt.figure(figsize=[.5 * 6.4, .5 * 4.9])
plt.scatter(x, y, s=20, color='k')
plt.plot(t, model.predict(t.reshape(-1, 1)), color='C3',
         label='Fitted model')

plt.plot(t, f(t), 'k--', label='Best possible fit\n$\\approx$generative process')
style_figs.no_axis()
plt.ylim(-1.25, 2.5)
plt.legend(loc='upper right', borderaxespad=0, borderpad=0,
           labelspacing=.6, fontsize=16)
plt.subplots_adjust(top=1)


# More detailed legend
plt.savefig('polynomial_overfit_simple_legend.svg', facecolor='none', edgecolor='none')

# %%
# Underfit settings

model = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
model.fit(x.reshape(-1, 1), y)

plt.figure(figsize=[.5 * 6.4, .5 * 4.9])
plt.scatter(x, y, s=20, color='k')
plt.plot(t, model.predict(t.reshape(-1, 1)), color='C0',
         label='Fitted model\n$\\approx$best possible fit')

plt.plot(t, f(t), 'k--', label='Generative process')
style_figs.no_axis()
plt.ylim(-1.25, 2.5)
plt.legend(loc='upper right', borderaxespad=0, borderpad=0,
           labelspacing=.4, fontsize=16)
plt.subplots_adjust(top=1)

plt.savefig('polynomial_overfit_assymptotic.svg', facecolor='none', edgecolor='none')


# %%
# Validation curves
from sklearn import model_selection
plt.figure()

param_range = np.arange(1, 20)

train_scores, test_scores = model_selection.validation_curve(
    model, x[::2].reshape((-1, 1)), y[::2],
    param_name='polynomialfeatures__degree',
    param_range=param_range,
    cv=model_selection.ShuffleSplit(n_splits=20))

plt.plot(param_range, -np.mean(test_scores, axis=1), 'k',
         label='Generalization error')
plt.plot(param_range, -np.mean(train_scores, axis=1), 'k--',
         label='Training error')

ax = plt.gca()
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)

plt.ylim(ymax=.05)

plt.legend(loc='center')

plt.yticks(())
plt.ylabel('Error')
plt.xlabel('Polynomial degree')
plt.subplots_adjust(left=.07, bottom=.18, top=.99, right=.99)

plt.savefig('polynomial_validation_curve.svg', facecolor='none',
            edgecolor='none')

# %%
# Learning curves
rng = np.random.RandomState(0)
x = 2 * rng.rand(100 * N_SAMPLES) - 1

y = f(x) + .4 * rng.normal(size=100 * N_SAMPLES)

X = x.reshape((-1, 1))

np.random.seed(42)

plt.figure()

def savefig(name):
    " Ajust layout, and then save"
    ax = plt.gca()
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    plt.ylim(-.65, .15)
    plt.xlim(train_sizes.min(), train_sizes.max())
    plt.xticks((100, 1000), ('100', '1000'), size=13)
    plt.yticks(())

    plt.ylabel('Error')
    plt.xlabel('Number of samples')
    plt.subplots_adjust(left=.07, bottom=.16, top=.99, right=.99)
    plt.savefig(name, edgecolor='none', facecolor='none')


# Degree 9
model = make_pipeline(PolynomialFeatures(degree=9), LinearRegression())
train_sizes, train_scores, test_scores = model_selection.learning_curve(
    model, X, y, cv=model_selection.ShuffleSplit(n_splits=20),
    train_sizes=np.logspace(-2.5, -.3, 30))
test_plot = plt.semilogx(train_sizes, -np.mean(test_scores, axis=1),
                            label='9',
                            color='C3')
savefig('polynomial_learning_curve_0.svg')
train_plot = plt.semilogx(train_sizes, -np.mean(train_scores, axis=1), '--',
                            color='C3')

leg1 = plt.legend(['Generalization error', 'Training error'],
                    loc='upper right', borderaxespad=-.2)
savefig('polynomial_learning_curve_1.svg')

# Degree 1
model = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
train_sizes, train_scores, test_scores = model_selection.learning_curve(
    model, X, y, cv=model_selection.ShuffleSplit(n_splits=20),
    train_sizes=np.logspace(-2.5, -.3, 30))
test_plot = plt.semilogx(train_sizes, -np.mean(test_scores, axis=1),
                            label='1',
                            color='C0')
train_plot = plt.semilogx(train_sizes, -np.mean(train_scores, axis=1), '--',
                            color='C0')


plt.legend(loc='right',
           title='Degree of polynomial', ncol=2)
plt.gca().add_artist(leg1)
savefig('polynomial_learning_curve.svg')

# %%
# Simple figure to demo overfit: with our polynomial data

N_SAMPLES = 50

rng = np.random.RandomState(0)
x = 2 * rng.rand(N_SAMPLES) - 1

y = f(x) + .4 * rng.normal(size=N_SAMPLES)

x_test = 2 * rng.rand(10*N_SAMPLES) - 1

y_test = f(x_test) + .4 * rng.normal(size=10*N_SAMPLES)

plt.figure(figsize=(.8*4, .8*3), facecolor='none')
plt.clf()
ax = plt.axes([.1, .1, .9, .9])

from sklearn import linear_model
# Create linear regression object
regr = linear_model.LinearRegression()
regr.fit(x.reshape((-1, 1)), y)

plt.scatter(x, y,  color='k', s=9)

plt.plot([-1, 1], regr.predict([[-1, ], [1, ]]),
        linewidth=3)

plt.axis('tight')
plt.xlim(-1, 1)
ymin, ymax = plt.ylim()
style_figs.light_axis()
plt.ylabel('y', size=16, weight=600)
plt.xlabel('x', size=16, weight=600)

plt.savefig('ols_simple.svg', facecolor='none', edgecolor='none')

plt.scatter(x_test, y_test,  color='C1', s=9, zorder=20, alpha=.4)
plt.xlim(-1, 1)
plt.ylim(ymin, ymax)

plt.savefig('ols_simple_test.svg', facecolor='none', edgecolor='none')


# %%
# Plot cubic splines
plt.clf()
ax = plt.axes([.1, .1, .9, .9])

from scipy import interpolate
f = interpolate.interp1d(x, y,
                         kind="quadratic",
                         bounds_error=False, fill_value="extrapolate")
plt.scatter(x, y,  color='k', s=9, zorder=20)
x_spline = np.linspace(-1, 1, 600)
y_spline = f(x_spline)
plt.plot(x_spline, y_spline, linewidth=3)

plt.axis('tight')
plt.xlim(-1, 1)
plt.ylim(ymin, ymax)

style_figs.light_axis()

plt.ylabel('y', size=16, weight=600)
plt.xlabel('x', size=16, weight=600)


plt.savefig('splines_cubic.svg', facecolor='none', edgecolor='none')

plt.scatter(x_test, y_test,  color='C1', s=9, zorder=20, alpha=.4)
plt.xlim(-1, 1)
plt.ylim(ymin, ymax)

plt.savefig('splines_cubic_test.svg', facecolor='none', edgecolor='none')

# %%
# Simple figure to demo overfit: with linearly-generated data

N_SAMPLES = 50

rng = np.random.RandomState(0)
x = 2 * rng.rand(N_SAMPLES) - 1

y = regr.coef_ * x + regr.intercept_ + .4 * rng.normal(size=N_SAMPLES)

x_test = 2 * rng.rand(10*N_SAMPLES) - 1

y_test = regr.coef_ * x_test + regr.intercept_ + .4 * rng.normal(size=10*N_SAMPLES)

plt.figure(figsize=(.8*4, .8*3), facecolor='none')
plt.clf()
ax = plt.axes([.1, .1, .9, .9])

from sklearn import linear_model
# Create linear regression object
regr = linear_model.LinearRegression()
regr.fit(x.reshape((-1, 1)), y)

plt.scatter(x, y,  color='k', s=9)

plt.plot([-1, 1], regr.predict([[-1, ], [1, ]]),
        linewidth=3)

plt.axis('tight')
plt.xlim(-1, 1)
plt.ylim(ymin, ymax)
style_figs.light_axis()
plt.ylabel('y', size=16, weight=600)
plt.xlabel('x', size=16, weight=600)

plt.savefig('linear_ols.svg', facecolor='none', edgecolor='none')

plt.scatter(x_test, y_test,  color='C1', s=9, zorder=20, alpha=.4)
plt.xlim(-1, 1)
plt.ylim(ymin, ymax)

plt.savefig('linear_ols_test.svg', facecolor='none', edgecolor='none')


# %%
# Plot cubic splines
plt.clf()
ax = plt.axes([.1, .1, .9, .9])

from scipy import interpolate
f = interpolate.interp1d(x, y,
                         kind="quadratic",
                         bounds_error=False, fill_value="extrapolate")
plt.scatter(x, y,  color='k', s=9, zorder=20)
x_spline = np.linspace(-1, 1, 600)
y_spline = f(x_spline)
plt.plot(x_spline, y_spline, linewidth=3)

plt.axis('tight')
plt.xlim(-1, 1)
plt.ylim(ymin, ymax)

style_figs.light_axis()

plt.ylabel('y', size=16, weight=600)
plt.xlabel('x', size=16, weight=600)


plt.savefig('linear_splines.svg', facecolor='none', edgecolor='none')

plt.scatter(x_test, y_test,  color='C1', s=9, zorder=20, alpha=.4)
plt.xlim(-1, 1)
plt.ylim(ymin, ymax)

plt.savefig('linear_splines_test.svg', facecolor='none', edgecolor='none')



