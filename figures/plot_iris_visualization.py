"""
Some simple visualizations on the iris data.
"""

import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
import style_figs

iris = datasets.load_iris()

# Plot the histograms of each class for each feature


X = iris.data
y = iris.target
for x, feature_name in zip(X.T, iris.feature_names):
    plt.figure(figsize=(2.5, 2))
    patches = list()
    for this_y, target_name in enumerate(iris.target_names):
        patch = plt.hist(x[y == this_y],
                         bins=np.linspace(x.min(), x.max(), 20),
                         label=target_name)
        patches.append(patch[-1][0])
    style_figs.light_axis()
    feature_name = feature_name.replace(' ', '_')
    feature_name = feature_name.replace('(', '')
    feature_name = feature_name.replace(')', '')
    plt.savefig('iris_{}_hist.svg'.format(feature_name))

plt.figure(figsize=(6, .25))
plt.legend(patches, iris.target_names, ncol=3, loc=(0, -.37),
           borderaxespad=0)
style_figs.no_axis()
plt.savefig('legend_irises.svg')


