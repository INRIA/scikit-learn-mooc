# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,markdown_files//md,python_scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# # Unsupervised Learning Part 1 -- Transformation
#

# %% [markdown]
# Many instances of unsupervised learning, such as dimensionality reduction, manifold learning, and feature extraction, find a new representation of the input data without any additional input. (In contrast to supervised learning, usnupervised algorithms don't require or consider target variables like in the previous classification and regression examples). 
#
# <img src="figures/unsupervised_workflow.svg" width="100%">
#
# A very basic example is the rescaling of our data, which is a requirement for many machine learning algorithms as they are not scale-invariant -- rescaling falls into the category of data pre-processing and can barely be called *learning*. There exist many different rescaling technques, and in the following example, we will take a look at a particular method that is commonly called "standardization." Here, we will recale the data so that each feature is centered at zero (mean = 0) with unit variance (standard deviation = 0).
#
# For example, if we have a 1D dataset with the values [1, 2, 3, 4, 5], the standardized values are
#
# - 1 -> -1.41
# - 2 -> -0.71
# - 3 -> 0.0
# - 4 -> 0.71
# - 5 -> 1.41
#
# computed via the equation $x_{standardized} = \frac{x - \mu_x}{\sigma_x}$,
# where $\mu$ is the sample mean, and $\sigma$ the standard deviation, respectively.
#
#
#
#
#

# %%
ary = np.array([1, 2, 3, 4, 5])
ary_standardized = (ary - ary.mean()) / ary.std()
ary_standardized

# %% [markdown]
# Although standardization is a most basic preprocessing procedure -- as we've seen in the code snipped above -- scikit-learn implements a `StandardScaler` class for this computation. And in later sections, we will see why and when the scikit-learn interface comes in handy over the code snippet we executed above.  
#
# Applying such a preprocessing has a very similar interface to the supervised learning algorithms we saw so far.
# To get some more practice with scikit-learn's "Transformer" interface, let's start by loading the iris dataset and rescale it:
#

# %%
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
print(X_train.shape)

# %% [markdown]
# The iris dataset is not "centered" that is it has non-zero mean and the standard deviation is different for each component:
#

# %%
print("mean : %s " % X_train.mean(axis=0))
print("standard deviation : %s " % X_train.std(axis=0))

# %% [markdown]
# To use a preprocessing method, we first import the estimator, here StandardScaler and instantiate it:
#     

# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# %% [markdown]
# As with the classification and regression algorithms, we call ``fit`` to learn the model from the data. As this is an unsupervised model, we only pass ``X``, not ``y``. This simply estimates mean and standard deviation.

# %%
scaler.fit(X_train)

# %% [markdown]
# Now we can rescale our data by applying the ``transform`` (not ``predict``) method:

# %%
X_train_scaled = scaler.transform(X_train)

# %% [markdown]
# ``X_train_scaled`` has the same number of samples and features, but the mean was subtracted and all features were scaled to have unit standard deviation:

# %%
print(X_train_scaled.shape)

# %%
print("mean : %s " % X_train_scaled.mean(axis=0))
print("standard deviation : %s " % X_train_scaled.std(axis=0))

# %% [markdown]
# To summarize: Via the `fit` method, the estimator is fitted to the data we provide. In this step, the estimator estimates the parameters from the data (here: mean and standard deviation). Then, if we `transform` data, these parameters are used to transform a dataset. (Please note that the transform method does not update these parameters).

# %% [markdown]
# It's important to note that the same transformation is applied to the training and the test set. That has the consequence that usually the mean of the test data is not zero after scaling:

# %%
X_test_scaled = scaler.transform(X_test)
print("mean test data: %s" % X_test_scaled.mean(axis=0))

# %% [markdown]
# It is important for the training and test data to be transformed in exactly the same way, for the following processing steps to make sense of the data, as is illustrated in the figure below:

# %%
from figures import plot_relative_scaling
plot_relative_scaling()

# %% [markdown]
# There are several common ways to scale the data. The most common one is the ``StandardScaler`` we just introduced, but rescaling the data to a fix minimum an maximum value with ``MinMaxScaler`` (usually between 0 and 1), or using more robust statistics like median and quantile, instead of mean and standard deviation (with ``RobustScaler``), are also useful.

# %%
from figures import plot_scaling
plot_scaling()

# %% [markdown]
# Principal Component Analysis
# ============================

# %% [markdown]
# An unsupervised transformation that is somewhat more interesting is Principal Component Analysis (PCA).
# It is a technique to reduce the dimensionality of the data, by creating a linear projection.
# That is, we find new features to represent the data that are a linear combination of the old data (i.e. we rotate it). Thus, we can think of PCA as a projection of our data onto a *new* feature space.
#
# The way PCA finds these new directions is by looking for the directions of maximum variance.
# Usually only few components that explain most of the variance in the data are kept. Here, the premise is to reduce the size (dimensionality) of a dataset while capturing most of its information. There are many reason why dimensionality reduction can be useful: It can reduce the computational cost when running learning algorithms, decrease the storage space, and may help with the so-called "curse of dimensionality," which we will discuss in greater detail later.
#
# To illustrate how a rotation might look like, we first show it on two-dimensional data and keep both principal components. Here is an illustration:
#
#

# %%
from figures import plot_pca_illustration
plot_pca_illustration()

# %% [markdown]
# Now let's go through all the steps in more detail:
# We create a Gaussian blob that is rotated:

# %%
rnd = np.random.RandomState(5)
X_ = rnd.normal(size=(300, 2))
X_blob = np.dot(X_, rnd.normal(size=(2, 2))) + rnd.normal(size=2)
y = X_[:, 0] > 0
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y, linewidths=0, s=30)
plt.xlabel("feature 1")
plt.ylabel("feature 2");

# %% [markdown]
# As always, we instantiate our PCA model. By default all directions are kept.

# %%
from sklearn.decomposition import PCA
pca = PCA()

# %% [markdown]
# Then we fit the PCA model with our data. As PCA is an unsupervised algorithm, there is no output ``y``.

# %%
pca.fit(X_blob)

# %% [markdown]
# Then we can transform the data, projected on the principal components:

# %%
X_pca = pca.transform(X_blob)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, linewidths=0, s=30)
plt.xlabel("first principal component")
plt.ylabel("second principal component");

# %%
pca = PCA(n_components=1).fit(X_blob)

# %%
X_blob.shape

# %%
pca.transform(X_blob).shape

# %% [markdown]
# On the left of the plot you can see the four points that were on the top right before. PCA found fit first component to be along the diagonal, and the second to be perpendicular to it. As PCA finds a rotation, the principal components are always at right angles ("orthogonal") to each other.

# %% [markdown]
# Dimensionality Reduction for Visualization with PCA
# -------------------------------------------------------------
# Consider the digits dataset. It cannot be visualized in a single 2D plot, as it has 64 features. We are going to extract 2 dimensions to visualize it in, using the example from the sklearn examples [here](http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html)

# %%
from figures import digits_plot

digits_plot()

# %% [markdown]
# Note that this projection was determined *without* any information about the
# labels (represented by the colors): this is the sense in which the learning
# is **unsupervised**.  Nevertheless, we see that the projection gives us insight
# into the distribution of the different digits in parameter space.

# %% [markdown]
# <div class="alert alert-success">
#     <b>EXERCISE</b>:
#      <ul>
#       <li>
#         Visualize the iris dataset using the first two principal components, and compare this visualization to using two of the original features.
#       </li>
#     </ul>
# </div>

# %%
# # %load solutions/07A_iris-pca.py
