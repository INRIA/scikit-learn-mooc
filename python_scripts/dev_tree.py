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
import numpy as np
import matplotlib.pyplot as plt


def plot_decision_function(X, y, clf):
    """Plot the boundary of the decision function of a classifier."""
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
# ## Go into details in the partitioning mechanism
#
# We saw in the previous section that a tree uses a mechanism to partition the
# feature space only considering a feature at a time. We will now go into
# details regarding the split.
#
# We will focus on a single feature and check how a split will be defined.

# %%
single_feature = X_train["Culmen Length (mm)"]

# %% [markdown]
# We can check once more what is the distribution of this feature

# %%
for klass in y_train.unique():
    mask_penguin_species = y_train == klass
    plt.hist(
        single_feature[mask_penguin_species], alpha=0.7,
        label=f'{klass}', density=True
    )
plt.legend()
plt.xlabel(single_feature.name)
_ = plt.ylabel('Class probability')

# %% [markdown]
# On this graph, we can see that we can easily separate the Adelie specie from
# the other species. Instead of the distribution, we can alternatively have
# a representation of all samples.

# %%
df = pd.concat(
    [single_feature, y_train,
     pd.Series([""] * y_train.size, index=single_feature.index, name="")],
    axis=1,
)
_ = sns.swarmplot(x=single_feature.name, y="", hue=y_train.name, data=df)

# %% [markdown]
# Finding a split is then equivalent to find a threshold value which will be
# used to separate the classes. To give an example, we will pick a random
# threshold value and check what would be the quality of the split.

# %%
random_indice = np.random.choice(single_feature.index)
threshold_value = single_feature.loc[random_indice]

_, ax = plt.subplots()
_ = sns.swarmplot(
    x=single_feature.name, y="", hue=y_train.name, data=df, ax=ax
)
ax.axvline(threshold_value, linestyle="--", color="black")
_ = ax.set_title(f"Random threshold value: {threshold_value} mm")

# %% [markdown]
# A random selection does not ensure that we pick up the best threshold which
# will separate the most the different species. Here, the intuition is that
# we need to find a threshold that best divide the Adelie class from other
# classes. A threshold around 42 mm would be ideal. Once this split defined,
# we could specify that the sample < 42 mm would belong to the class Adelie and
# the samples > 42 mm would belong to class the most probable (the most
# represented) between the Gentoo and the Chinstrap. In this case, it seems to
# be the Gentoo, which is in-line with what we observed earlier when fitting a
# `DecisionTreeClassifier` with a `max_depth=1`.

# %%
threshold_value = 42

_, ax = plt.subplots()
_ = sns.swarmplot(
    x=single_feature.name, y="", hue=y_train.name, data=df, ax=ax
)
ax.axvline(threshold_value, linestyle="--", color="black")
_ = ax.set_title(f"Manual threshold value: {threshold_value} mm")

# %% [markdown]
# Intuitively, we expect the best possible threshold to be around this value
# (42 mm) because it is the split leading the least amount of error that we
# would make. Thus, if we want to automatically find such a threshold, we would
# need a way to evaluate the goodness (or pureness) of a given threshold. This
# evaluation is based on a tryptic of statistical measures, namely,
# probabilities, entropy, and information gain.
#
# ### Probabilities
#
# First, we will investigate how probabilities can help us evaluating the
# goodness of a split. We will first defined as if we have found a split for
# the threshold 42 mm. For this threshold, we will then divide our data into
# 2 sub-groups: 1 group for samples < 42 mm and 1 group for samples >= 42 mm.
# Then, we will store the class label for these samples.

# %%
threshold_value = 42
mask_below_threshold = single_feature < threshold_value
labels_below_threshold = y_train[mask_below_threshold]
labels_above_threshold = y_train[~mask_below_threshold]

# %% [markdown]
# If we want to investigate the goodness of the split, we will to check the
# labels frequency of each side of the split.

# %%
labels_below_threshold.value_counts()

# %%
labels_above_threshold.value_counts()


# %% [markdown]
# As we could have assess previously, the partition defined by < 42 mm has
# mainly Adelie penguin and only 2 samples which we could considered
# misclassified. However on the partition >= 42 mm, we cannot differentiate
# Gentoo and Chinstrap (while they are almost twice more Gentoo).
#
# These frequencies are useful but they cannot be easily compared since they
# are not a normalized quantity. Thus, we could normalize the frequencies by
# the total number of samples on the partition and we would obtain the
# probability to be a certain class on this partition.

# %%
def compute_probability_partition(labels):
    return labels.value_counts(normalize=True).sort_index()


probability_below_threshold = compute_probability_partition(
    labels_below_threshold
)
probability_above_threshold = compute_probability_partition(
    labels_above_threshold
)

print(f"Probability for partition below the threshold: \n"
      f"{probability_below_threshold}")
print(f"Probability for partition above the threshold: \n"
      f"{probability_above_threshold}")


# %% [markdown]
# Probabilities allow us to quantify the amount of sample in each partition.
# However, we have one probability per class and one would wish to have a
# single value to know the pureness of a partition or in other words, is the
# partition composed of a single class or is it a mix of all classes.
#
# ### Entropy
#
# The entropy is one of the statistics that can help in this regard. It will
# combine the different probabilities such as:
# $H(X) = - \sum_{k=1}^{K} p(X_k) \log p(X_k)$
#
# For a binary problem, the entropy function for one of the class can be
# depicted as:
#
# ![title](https://upload.wikimedia.org/wikipedia/commons/2/22/Binary_entropy_plot.svg)
#
# Therefore, the entropy will be maximum when the proportion of sample from
# each class will be equal and minimum when only samples for a single class
# is present.
#
# To conclude, one search to minimize the entropy in a partition.

# %%
def entropy(labels):
    from scipy import stats
    probabilities = compute_probability_partition(labels)
    return stats.entropy(probabilities)


entropy_below_threshold = entropy(labels_below_threshold)
entropy_above_threshold = entropy(labels_above_threshold)

print(f"Entropy for partition below the threshold: \n"
      f"{entropy_below_threshold}")
print(f"Entropy for partition above the threshold: \n"
      f"{entropy_above_threshold}")


# %% [markdown]
# In our case, we can see that the entropy in the partition < 42 mm is close to
# 0 meaning that this partition is "pure" and contain a single class while
# the partition >= 42 mm is much higher due to the fact that 2 of the classes
# are still mixed.
#
# Now, we are able to access the quality of each partition. However, the
# ultimate goal is to evaluate the quality of the split and thus combine both
# measure of entropy to obtain a single statistic.
#
# ### Information gain
#
# This statistic is known as the information gain. It combines the entropy of
# the different partitions to give us a single statistic qualifying the quality
# of the split. The information gain is defined as the difference of the
# entropy before the split and the sum of the entropies of the partition each
# normalized by the frequencies of class samples on each partition. The goal is
# to maximize the information gain.
#
# We will define a function to compute the information gain given the different
# partitions.

# %%
def information_gain(labels_below_threshold, labels_above_threshold):
    # compute the entropies in the different partitions
    entropy_below_threshold = entropy(labels_below_threshold)
    entropy_above_threshold = entropy(labels_above_threshold)
    entropy_parent = entropy(
        pd.concat([labels_below_threshold, labels_above_threshold])
    )

    # compute the normalized entropies
    n_samples_below_threshold = labels_below_threshold.size
    n_samples_above_threshold = labels_above_threshold.size
    n_samples_parent = n_samples_below_threshold + n_samples_above_threshold

    normalized_entropy_below_threshold = (
        (n_samples_below_threshold / n_samples_parent) *
        entropy_below_threshold
    )
    normalized_entropy_above_threshold = (
        (n_samples_above_threshold / n_samples_parent) *
        entropy_above_threshold
    )

    # compute the information gain
    return (entropy_parent -
            normalized_entropy_below_threshold -
            normalized_entropy_above_threshold)


information_gain(labels_below_threshold, labels_above_threshold)

# %% [markdown]
# Now that we are able to quantify the quality of a split, we can compute the
# information gain for all the different possible splits.

# %%
splits_information_gain = []
possible_thresholds = np.sort(single_feature.unique())[1:-1]
for threshold_value in possible_thresholds:
    mask_below_threshold = single_feature < threshold_value
    labels_below_threshold = y_train.loc[mask_below_threshold]
    labels_above_threshold = y_train.loc[~mask_below_threshold]
    splits_information_gain.append(
        information_gain(labels_below_threshold, labels_above_threshold)
    )

plt.plot(possible_thresholds, splits_information_gain)
plt.xlabel(single_feature.name)
_ = plt.ylabel("Information gain")

# %% [markdown]
# As previously mentioned, we would like to find the threshold value maximizing
# the information gain.

# %%
best_threshold_indice = np.argmax(splits_information_gain)
best_threshold_value = possible_thresholds[best_threshold_indice]

_, ax = plt.subplots()
ax.plot(possible_thresholds, splits_information_gain)
ax.set_xlabel(single_feature.name)
ax.set_ylabel("Information gain")
ax.axvline(best_threshold_value, color="tab:orange", linestyle="--")
ax.set_title(f"Best threshold: {best_threshold_value} mm")

# %% [markdown]
# By making this brute-force search, we find that the threshold maximizing the
# information gain is 43.3 mm. We can check if we found something similar when
# using the `DecisionTreeClassifier` earlier.

# %%
from sklearn.tree import plot_tree

tree = DecisionTreeClassifier(criterion="entropy", max_depth=1)
tree.fit(single_feature.to_frame(), y_train)
_ = plot_tree(tree)


# %% [markdown]
# We can observe that the implementation in scikit-learn is giving something
# very similar: 43.25 mm. The slight difference are only due to some low-level
# implementation details.
#
# Once a split done, the data will be partitioned and we can restart the
# process or partitioning on each subset. In the above example, it corresponds
# to increase the `max_depth` parameter.
#
# ## How prediction works?
#
# We showed the way a tree is constructed. However, we did not explain how and
# what will be predicted from the tree.
#
# We can first recall the structure of the tree that we just fitted.

# %%
_ = plot_tree(tree)

# %% [markdown]
# So the threshold value is 43.25 so we can check which classes we are going
# to predict for a value above and below this threshold

# %%
print(f"The class predicted for a value below the threshold is: "
      f"{tree.predict([[35]])}")
print(f"The class predicted for a value above the threshold is: "
      f"{tree.predict([[45]])}")

# %% [markdown]
# We predict an Adelie penguin for value below the threshold which is not
# surprising since this partition was almost pure. In the case that it is not
# as obvious, we predicted the Gentoo penguin. Indeed, we predict the class the
# most probable (i.e. coming from the probabilities that we computed above).
