# %% [markdown]
# # Decision tree in depth
#
# In this notebook, we will go into details on the internal algorithm of the
# decision tree. In this regard, we will use a simple dataset composed of only
# 2 features. We will illustrate how the space partioning along each feature
# occur allowing to obtain a final decision.
#
# ## Presentation of the dataset
#
# We use the
# [Palmer penguins dataset](https://allisonhorst.github.io/palmerpenguins/).
# This dataset is composed of penguins record and the final aim is to identify
# from which specie a penguin belongs to.
#
# The penguins belongs to three different species: Adelie, Gentoo, and
# Chinstrap. Here is an illustration of the three different species:
#
# ![Image of penguins](https://github.com/allisonhorst/palmerpenguins/raw/master/man/figures/lter_penguins.png)
#
# Here, we will only use a subset of the original features based on the
# penguin's culmen. You can know more about the penguin's culmen in the below
# illustration:
#
# ![Image of culmen](https://github.com/allisonhorst/palmerpenguins/raw/master/man/figures/culmen_depth.png)

# %%
import pandas as pd

data = pd.read_csv("../datasets/penguins.csv")

# select the features of interest
culmen_columns = ["Culmen Length (mm)", "Culmen Depth (mm)"]
target_column = "Species"

data = data[culmen_columns + [target_column]]
data[target_column] = data[target_column].str.split().str[0]

# %% [markdown]
# Let's check the dataset more into details

# %%
data.info()

# %% [markdown]
# We can observe that they are 2 missing records in this dataset and for a sake
# of simplicity, we will drop the records corresponding to these 2 samples.

# %%
data = data.dropna()
data.info()

# %% [markdown]
# We split the data and target into 2 different variables and divide it into
# a training and testing set.

# %%
from sklearn.model_selection import train_test_split

X, y = data[culmen_columns], data[target_column]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=0,
)

# %% [markdown]
# Before to go into details in the decision tree algorithm, let's have a visual
# inspection of the distribution of the culmen length and width depending of
# the penguin's species.

# %%
import seaborn as sns

_ = sns.pairplot(data=data, hue="Species")

# %% [markdown]
# Focusing on the diagonal plot on the pairplot and thus the distribution of
# each individual feature, we can build some intuitions:
#
# * The Adelie specie can be separated from the Gentoo and Chinstrap species
#   using the culmen length;
# * The Gentoo specie can be separated from the Adelie and Chinstrap species
#   using the culmen depth.
#
# ## Build an intuition on how decision trees work
#
# We saw in a previous notebook that a linear classifier will find a separation
# defined as a combination of the input feature. In our 2-dimensional space, it
# means that a linear classifier will defined some oblique lines that best
# separate our classes. We define a function below that given a set of data
# point and a classifier will plot the decision boundaries learnt by the
# classifier.


# %%
def plot_decision_function(X, y, clf):
    """Plot the boundary of the decision function of a classifier."""
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import LabelEncoder

    clf.fit(X, y)

    # create a grid to evaluate all possibilities
    plot_step = 0.02
    feature_0_min, feature_0_max = (X.iloc[:, 0].min() - 1,
                                    X.iloc[:, 0].max() + 1)
    feature_1_min, feature_1_max = (X.iloc[:, 1].min() - 1,
                                    X.iloc[:, 1].max() + 1)
    xx, yy = np.meshgrid(
        np.arange(feature_0_min, feature_0_max, plot_step),
        np.arange(feature_1_min, feature_1_max, plot_step)
    )

    # compute the associated prediction
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = LabelEncoder().fit_transform(Z)
    Z = Z.reshape(xx.shape)

    # make the plot of the boundary and the data samples
    _, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.4)
    sns.scatterplot(
        data=pd.concat([X, y], axis=1),
        x=X.columns[0], y=X.columns[1], hue=y.name,
        ax=ax,
    )


# %% [markdown]
# Thus, for a linear classifier, we will obtain the following decision
# boundaries.

# %%
from sklearn.linear_model import LogisticRegression

linear_model = LogisticRegression()
plot_decision_function(X_train, y_train, linear_model)
linear_model.fit(X_train, y_train).score(X_test, y_test)

# %% [markdown]
# Thus, we see that the lines are a combination of the input features since
# they are not perpendicular a specific axis.
#
# In the contrary, decision tree will partition the space considering a single
# feature. Let's illustrate this behaviour.

# %%
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=1)
plot_decision_function(X_train, y_train, tree)

# %% [markdown]
# Thus, we see that the partition found by our decision tree was to separate
# data along the axis "Culmen Length", discarding the feature "Culmen Depth".
#
# However, such a split is not powerful enough to isolate our three species and
# our accuracy will be low compared to our linear model.

# %%
tree.fit(X_train, y_train).score(X_test, y_test)

# %% [markdown]
# This is not a surprise since we saw in the section above that a single
# feature will not help to separate the three species. However, from this
# previous analysis we saw that if we use both feature, we should get fairly
# good results in separating all 3 classes.
# Considering the mechanism of the decision tree, it means that we should
# repeat a partitioning on each rectangle that we created previously and we
# expect that this partitioning will happen considering the feature
# "Culmen Depth" this time.

# %%
tree.set_params(max_depth=2)
plot_decision_function(X_train, y_train, tree)

# %% [mark]
# As expected, the decision tree partionned using the "Culmen Depth" to
# identify correctly data that we did could distinguish with a single split.
# We see that our tree is more powerful with similar performance to our linear
# model.

# %%
tree.fit(X_train, y_train).score(X_test, y_test)

# %% [markdown]
# Now that we got the intuition that the algorithm of the decision tree
# corresponds to a successive partitioning of our feature space considering a
# feature at a time, we can focus on the details on how we can compute the best
# partition.
#
# ## Go in detail into the partitioning mechanism
#
