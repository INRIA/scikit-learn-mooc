# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import style_figs


# %%
# graph tree intro : https://docs.google.com/drawings/d/1gbYLXWpubn5CPudGKPMhETyLwDcvKS2yLiAzF1nIgFo/edit?usp=sharing

# %%
dataset = pd.read_csv("../datasets/penguins.csv").dropna(subset=["Body Mass (g)"])
dataset = dataset[[spe[:3] != 'Chi' for spe in dataset['Species']]]
color = ['C0' if (x[0] == "A") else 'C1' for x in dataset["Species"]]
plt.figure(figsize = (4,3))
style_figs.light_axis()
plt.ylabel('$x_1$', size=26, weight=600)
plt.xlabel('$x_0$', size=26, weight=600)
plt.xlim((2700, 6600))


# sns.scatterplot(x = 'Body Mass (g)', y= 'Culmen Depth (mm)', hue = 'Species', data = dataset,
#                legend = False)
plt.scatter(dataset['Body Mass (g)'], dataset['Culmen Depth (mm)'], c = color, 
            cmap=plt.cm.bwr, edgecolor='black')
x_sep = 4300
plt.plot([x_sep, x_sep], [13,22], 'r-.')
plt.savefig('tree2D_1split.svg', bbox_inches = 'tight')

# plt.plot([x_sep, 6600], [17.4, 17.4], 'r:')
plt.plot([2700, x_sep], [15, 15], 'r', ls = ':')
plt.savefig('tree2D_2split.svg', bbox_inches = 'tight')

plt.plot([x_sep, 6600], [17.4, 17.4], 'r:')
plt.plot([2700, x_sep], [15, 15], 'r', ls = ':')
plt.savefig('tree2D_3split.svg', bbox_inches = 'tight')

#plt.scatter(dataset['Body Mass (g)'], dataset['Island'], c =cc)

# %%
# Visual with the above figure:
# https://docs.google.com/drawings/d/1rINa_f_qxlIjsDpVv78hHqIgG5heheNHdnbGZJtBRvo/edit?usp=sharing
# https://docs.google.com/drawings/d/1pL1soD6ZHKoOiAf5Z0zIecB9vJUxqAtuVMgAGJA_A4Y/edit?usp=sharing
# https://docs.google.com/drawings/d/1rkAnu_2pkn-Dk-Jo8vhJZJWDBUtBjwOP0chk4JjdT0s/edit?usp=sharing

# %% [markdown]
# # Regression

# %%

from sklearn.tree import DecisionTreeRegressor

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plot the results
plt.figure(figsize = (4,3))
ax = plt.axes([.1, .1, 1.5, .9])
# plt.axis('tight')
style_figs.light_axis()

# plt.xlim((0,7))
plt.ylabel('y', size=16, weight=600)
plt.xlabel('x', size=16, weight=600)
plt.xlim((0,5))

plt.scatter(X, y, s=40, edgecolor="black",
            c="k", label = 'data               ')
plt.legend(loc = 1, bbox_to_anchor=(.8, .5, 0.5, 0.5))
plt.savefig('tree_regression1.svg', facecolor='none', edgecolor='none',
            bbox_inches = 'tight')
plt.plot(X_test + .05 * np.ones(X_test.shape), y_1, color="C0",
         label="depth=2", linewidth=4)
# sort both labels and handles by labels
handles, labels = ax.get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
ax.legend(handles, labels,loc = 1, bbox_to_anchor=(.8, .5, 0.5, 0.5))
plt.savefig('tree_regression2.svg', facecolor='none', edgecolor='none',
            bbox_inches = 'tight')

plt.plot(X_test, y_2, color="C1", label="depth=5", linewidth=4)
# sort both labels and handles by labels
handles, labels = ax.get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
ax.legend(handles, labels,loc = 1, bbox_to_anchor=(.8, .5, 0.5, 0.5))

plt.savefig('tree_regression3.svg', facecolor='none', edgecolor='none',
            bbox_inches = 'tight')

# %%
# Decision tree underfit / overfit

from sklearn.ensemble import RandomForestRegressor

# Create the dataset
rng = np.random.RandomState(1)
X = np.linspace(0, 6, 100)[:, np.newaxis]
noise = rng.normal(0, 0.4, X.shape[0])

y_true = np.sin(X).ravel() + np.sin(6 * X).ravel()
y = y_true + noise

new_X = np.linspace(0, 6, 400)[:, np.newaxis]
new_noise = rng.normal(0, 0.3, new_X.shape[0])
new_y = np.sin(new_X).ravel() + np.sin(6 * new_X).ravel() + new_noise

# Plot
l_max_depth = [4,6,10]
l_n_est = [1, 20, 300]

for idx, col, label in zip([0,1,2], ['C0', 'C1', 'C2'],
                                 ['underfit', 'fit', 'overfit']):

    plt.figure(figsize = (4,3))
    plt.axes([.1, .1, .9, .9])
    style_figs.light_axis()
    plt.ylabel('y', size=22, weight=600)
    plt.xlabel('x', size=22, weight=600)

    plt.scatter(X,y, color = 'k')
    # plt.plot(X, y_true, color = 'k', label = 'y_true')
    plt.scatter(new_X,new_y, color = 'lightgrey', alpha = .5,
                s = 14, edgecolor = 'grey')

#     reg = RandomForestRegressor(n_estimators = l_n_est[idx], max_depth = 5)
    reg = DecisionTreeRegressor(max_depth = l_max_depth[idx])
    reg.fit(X,y)
    y_pred = reg.predict(X)
    plt.plot(X, y_pred, color = col)
#     plt.savefig(f'../figures/rf_{label}.svg', facecolor='none', edgecolor='none')
    plt.savefig(f'../figures/dt_{label}.svg', facecolor='none', edgecolor='none')



# %%

# %%

# %% [markdown]
# # Bagging rergession

# %%
# Bagging regression

from sklearn.ensemble import BaggingRegressor

# Create the dataset
rng = np.random.RandomState(1)
X = np.linspace(0, 3, 100)[:, np.newaxis]
noise = rng.normal(0, 0.4, X.shape[0])

y_true = np.sin(X).ravel() + np.sin(6 * X).ravel()
y = y_true + noise

# plot data
plt.figure(figsize=(4,3))
plt.axes([.1, .1, .9, .9])
style_figs.light_axis()
plt.ylabel('y', size=22, weight=600)
plt.xlabel('x', size=22, weight=600)
plt.scatter(X, y, color = 'k')
plt.savefig('../figures/bagging_reg_data.svg', facecolor='none', edgecolor='none')


def plot_subsample_bagging(seed = 0, line_col = 'C0', plot_reg = False):
    reg = DecisionTreeRegressor()
    rng = np.random.RandomState(ii)
    indice_subsample = np.sort(rng.choice(range(X.shape[0]), size = 20,
                                                replace = False, ))
    reg.fit(X[indice_subsample],y[indice_subsample])
    y_pred = reg.predict(X[indice_subsample])
    plt.scatter(X, y, color = 'white', edgecolor = 'k')
    plt.scatter(X[indice_subsample], y[indice_subsample], color = 'black', edgecolor = 'k')
    if plot_reg:
        plt.plot(X[indice_subsample], y_pred, color = line_col)
    
    
# Plot 7 grey bagging
plt.figure(figsize = (24, 4))
for ii in range(6):
    plt.subplot(1,6+1,ii+1)
    style_figs.light_axis()
    
    plot_subsample_bagging(seed = ii)
    

plt.savefig('../figures/bagging_reg_grey.svg', facecolor='none', edgecolor='none')
 
# Plot 7 grey bagging + reg line
plt.figure(figsize = (24, 4))
for ii in range(6):
    plt.subplot(1,6+1,ii+1)
    style_figs.light_axis()
    
    plot_subsample_bagging(seed = ii, plot_reg = True)
    
plt.savefig('../figures/bagging_reg_grey_fitted.svg', facecolor='none', edgecolor='none')
 

# %%


plt.figure(figsize=(4,3))
plt.axes([.1, .1, .9, .9])
style_figs.light_axis()
plt.ylabel('y', size=22, weight=600)
plt.xlabel('x', size=22, weight=600)
plt.scatter(X, y, color = 'grey', edgecolor = 'k')

reg = BaggingRegressor(n_estimators = 10) 
# reg = RandomForestRegressor() # same fig
reg.fit(X,y)
y_pred = reg.predict(X)
plt.plot(X, y_pred, color = 'C0')
plt.savefig('../figures/bagging_reg_blue.svg', facecolor='none', edgecolor='none')


# %%
plt.figure(figsize=(4,3))
plt.axes([.1, .1, .9, .9])
style_figs.light_axis()

plt.scatter(X, y, color = 'k')

for ii in range(7):

    plot_subsample_bagging(seed = ii, line_col = 'lightgrey')

plt.ylabel('y', size=22, weight=600)
plt.xlabel('x', size=22, weight=600)
plt.scatter(X, y, color = 'grey', edgecolor = 'k')

reg = BaggingRegressor(n_estimators = 10)
reg.fit(X,y)
y_pred = reg.predict(X)
plt.plot(X, y_pred, color = 'C0')
plt.savefig('../figures/bagging_reg_blue_grey.svg', facecolor='none', edgecolor='none')


# %%

# %%

# %% [markdown]
# # Bagging Classification

# %%
# tree drawing for bagging : 
# https://docs.google.com/drawings/d/1u5GrZTnIWHb3NCwMmuP-9iQpUfyDz0PHbhCF8X8mNlI/edit?usp=sharing
# https://docs.google.com/drawings/d/1Aj4FqGgkwD7M-f4hh1ZRK34YCWmKtmaYbuQ7WiJ-K8I/edit?usp=sharing
# https://docs.google.com/drawings/d/1KZHuvhQOueMXw1pVXS8rgdSKBB-wYcWblZYqtLlTts4/edit?usp=sharing
def plot_init(col_c = 'C0', col_s = 'C1', size = 90):
    
    circle = [[1,1], [1,3], [1.5,2], [1.4, 4.4], [.5,6], [2, 5.6]]
    circle2 = [[3, 6.5], [4, 5.5], [5, 5.8]]

    square = [[3,1], [3,4], [3.5, 2], [5,4.5], [5,1], [6.2, 3], [4.8, 2.8]]
    square2 = [[6,6]]
    
    plt.xlim((0,7))
    plt.ylim((0,7))

    plt.scatter([xy[0] for xy in circle], [xy[1] for xy in circle], c = col_c,
                edgecolor = 'k', s = size)
    plt.scatter([xy[0] for xy in circle2], [xy[1] for xy in circle2], c = col_c,
                edgecolor = 'k', s = size)

    plt.scatter([xy[0] for xy in square], [xy[1] for xy in square], marker = 's', c = col_s,
                edgecolor = 'k', s = size)
    plt.scatter([xy[0] for xy in square2], [xy[1] for xy in square2], marker = 's', c = col_s,
                edgecolor = 'k', s = size)
    
def plot_bagging(circle, square, new_point = False):
    plt.xlim((0,7))
    plt.ylim((0,7))
    plt.scatter([xy[0] for xy in circle], [xy[1] for xy in circle], c = 'C0',
                edgecolor = 'k', s = 160)
    plt.scatter([xy[0] for xy in square], [xy[1] for xy in square], marker = 's', c = 'C1',
                edgecolor = 'k', s = 160)
    if new_point:
        plt.scatter([3], [5], marker = 'X', c = 'grey', s = 350)

def plot_bagging3(plot_line = False, new_point = False):
    col_sep = 'grey'
    
    # 1
    plt.subplot(1,3,1)
    style_figs.light_axis()
    circle = [[1,1], [1,3], [1.5,2], [5, 5.8]]
    square = [[3,1], [3,4], [3.5, 2], [5,1]]
    plot_bagging(circle, square, new_point = new_point)
    if plot_line:
        plt.plot([2.3, 2.3], [0,7], col_sep, ls = '--')

    # 2
    plt.subplot(1, 3 ,2)
    style_figs.light_axis()
    circle = [[.5,6], [2, 5.6], [4, 5.5], [5, 5.8]]
    square = [[3,1], [3.5, 2],  [6.2, 3], [4.8, 2.8]]
    plot_bagging(circle, square, new_point = new_point)
    if plot_line:
        plt.plot([0, 7], [4, 4], col_sep, ls = '--')

    # 3
    plt.subplot(1, 3 ,3)
    style_figs.light_axis()
    circle = [[1,1],  [1.5,2], [1.4, 4.4],  [3, 6.5], [4, 5.5]]
    square = [[3,1], [5,4.5], [5,1], [6.2, 3]]

    if plot_line:
        plt.plot([4.5, 4.5], [0,7], col_sep, ls = '--')
    plot_bagging(circle, square, new_point = new_point)
    style_figs.light_axis()


    
plt.figure(figsize = (4, 3))
plot_init()
style_figs.light_axis()
plt.savefig('bagging0.svg', facecolor='none', edgecolor='none')


plt.figure(figsize = (12, 4))
plot_bagging3(plot_line = False, new_point = False)
plt.savefig('bagging.svg', facecolor='none', edgecolor='none')

plt.figure(figsize = (12, 4))
plot_bagging3(plot_line = True, new_point = False)
plt.savefig('bagging_line.svg', facecolor='none', edgecolor='none')

plt.figure(figsize = (12, 4))
plot_bagging3(plot_line = True, new_point = True)
plt.savefig('bagging_cross.svg', facecolor='none', edgecolor='none')

plt.figure(figsize=(4,3))
plot_init()
style_figs.light_axis()
plt.scatter([3], [5], marker = 'X', c = 'grey', s = 350)
plt.savefig('bagging0_cross.svg', facecolor='none', edgecolor='none')


# %% [markdown]
# # Boosting

# %% [markdown]
# # Boosting regression
#

# %%
# Create the dataset
rng = np.random.RandomState(1)
X = np.linspace(0, 3, 100)[:, np.newaxis]
noise = rng.normal(0, 0.2, X.shape[0])

y_true = np.sin(X).ravel() + np.sin(6 * X).ravel()
y = y_true + noise

# plot data
plt.figure(figsize=(4,3))
plt.axes([.1, .1, .9, .9])
style_figs.light_axis()
plt.ylabel('y', size=22, weight=600)
plt.xlabel('x', size=22, weight=600)
plt.scatter(X, y, color = 'k')
plt.savefig('../figures/boosting_reg_data.svg', facecolor='none', edgecolor='none')

# %%
# Boosting iteration 

from sklearn.ensemble import AdaBoostRegressor

def plot_boost_i(pos, i):
    plt.subplot(2,nb_subplot,pos+nb_subplot, label = str(i))
    style_figs.light_axis()
    y_loss = [int(x) for x in np.abs(y - y_pred)*100]
    plt.scatter(X, y, color = 'grey', edgecolor = 'k', s = y_loss)
    y_pred_i = adaboost[-1].estimators_[i-1].predict(X)
    plt.plot(X, y_pred_i, color = 'orange')
    
def plot_boost(pos, size = None):
    plt.subplot(2,nb_subplot,pos, label = str(i) + str(size))
    style_figs.light_axis()
    plt.scatter(X, y, color = 'grey', edgecolor = 'k', s = size)
    plt.plot(X, y_pred, color = 'blue')

adaboost = []
for i in range(1,100):
    adaboost.append(AdaBoostRegressor(n_estimators = i, random_state = 1).fit(X,y))

plt.figure(figsize=(24,8))

nb_subplot = 4
for i, pos in zip([1,2,4,20], range(1,nb_subplot+1)):
    y_pred = adaboost[i].predict(X)
    y_loss = [int(x) for x in np.abs(y - y_pred)*100]

    plot_boost(pos, size = None)
    plt.savefig('../figures/boosting/boosting_iter'+str(pos)+'.svg', facecolor='none', edgecolor='none')
    plot_boost(pos, size = y_loss)
    plt.savefig('../figures/boosting/boosting_iter_sized'+str(pos)+'.svg', facecolor='none', edgecolor='none')
    plot_boost_i(pos, i)
    plt.savefig('../figures/boosting/boosting_iter_orange'+str(pos)+'.svg', facecolor='none', edgecolor='none')
    

# %% [markdown]
# # Boosting Classification

# %%
# Link to tree drawing : 
# https://docs.google.com/drawings/d/1hlWjd74To_4Zg83SKdp2KvOfNhCKO6ty9eJLF_e_SaU/edit?usp=sharing


plt.figure(figsize = (4, 3))
plot_init()
style_figs.light_axis()
plt.savefig('boosting0.svg', facecolor='none', edgecolor='none')

col_sep = 'grey'

plt.figure(figsize = (4, 3))
plot_init(size = 80)
style_figs.light_axis()

plt.plot([2.3, 2.3], [0,7], col_sep, ls = '--')
circle2 = [[3, 6.5], [4, 5.5], [5, 5.8]]
plt.scatter([xy[0] for xy in circle2], [xy[1] for xy in circle2], c = 'C0',
            edgecolor = 'k', s = 350)
plt.savefig('boosting1.svg', facecolor='none', edgecolor='none')

plt.figure(figsize = (4, 3))
plot_init(size = 60)
style_figs.light_axis()

plt.plot([2.3, 2.3], [0,7], col_sep, ls = '--')
plt.plot([0, 7], [5, 5], col_sep, ls = '--')
square2 = [[6,6]]
plt.scatter([xy[0] for xy in square2], [xy[1] for xy in square2], marker = 's', c = 'C1',
            edgecolor = 'k', s = 450)
plt.savefig('boosting2.svg', facecolor='none', edgecolor='none')

plt.figure(figsize = (4, 3))
plot_init(size = 60)
style_figs.light_axis()

plt.plot([2.3, 2.3], [0,7], col_sep, ls = '--')
plt.plot([0, 7], [5, 5], col_sep, ls = '--')
plt.plot([5.5, 5.5], [0,7], col_sep, ls = '--')
plt.savefig('boosting3.svg', facecolor='none', edgecolor='none')
