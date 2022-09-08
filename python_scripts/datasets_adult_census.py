# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # The Adult census dataset
#
# This dataset is a collection of information related to a person. The
# prediction task is to predict whether a person is earning a salary above or
# below 50k USD/year.
#
# We explore this dataset in the first module's notebook "First look at our
# dataset". This provides a first intuition on how the data is structured. To
# avoid repeating the same information, here we explore with some more detail
# the relation between data imbalance and fairness.
#
# Remember that the data we use correspond to the 1994 US census that is
# available in [OpenML](http://openml.org/). A first thing to notice is that the
# information one can extract from it is outdated, not to mention that the
# variable names are somewhat controversial. We start as always by loading the
# dataset:

# %%
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")
target_name = "class"

# %% [markdown]
# The column named **class** is our target variable (i.e., the variable which we
# want to predict). The two possible classes are `"<=50K"` (low-revenue) and
# `">50K"` (high-revenue). In this dataset the target variable is slightly
# imbalanced, meaning there are more samples of the low-revenue class compared
# to the high revenue:

# %%
adult_census[target_name].value_counts()

# %% [markdown]
# The ratio of elements in the positive class over the number of elements in the
# negative class is called the **prevalence** and is a number between 0 and 1.
# In this case:

# %%
prevalence = (
    adult_census[target_name].value_counts()[1]
    / adult_census[target_name].value_counts()[0]
)
print(f"The prevalence of the dataset is: {prevalence:.3f}")

# %% [markdown]
# Notice that there is also an important imbalance on the data collection
# concerning the number of male/female samples.

# %%
adult_census["sex"].value_counts()

# %% [markdown]
# The class imbalance is even higher when the variable `"sex"` is accounted for:

# %%
adult_census.groupby("sex")[target_name].value_counts()

# %%
import seaborn as sns

_ = sns.catplot(x=target_name, hue="sex", kind="count", data=adult_census)

# %% [markdown]
# We can define a prevalence by group, or equivalently convert into percentage:

# %%
prevalence_female = (
    adult_census.groupby("sex")[target_name].value_counts()[1]
    / adult_census.groupby("sex")[target_name].value_counts()[0]
)
prevalence_male = (
    adult_census.groupby("sex")[target_name].value_counts()[3]
    / adult_census.groupby("sex")[target_name].value_counts()[2]
)
print(
    f"The percentage of 'female' samples earning a high revenue is {100*prevalence_female:.2f}%"
)
print(
    f"The percentage of 'male' samples earning a high revenue is {100*prevalence_male:.2f}%"
)

# %% [markdown]
# The proportion of 'female' samples earning a high revenue is less than the
# percentage of the equivalent quantity for the 'male' samples. The question is:
# Does the inequity come from the data-collection mechanism or the
# data-generating process?
#
# Let's imagine that we want to train a model on this basis to predict whether a
# person will be able to pay a loan and use this information to deside it's
# approval. Such decision will strongly impact their lives: buy a property or
# start a business, which may translate into economic stability and
# independence. Then a correct answer for this question is crucial.
#
# Before exploring different classifiers and metrics to evaluate them, we can go
# deeper into the data exploration and seek for an answer in the data-generating
# distributions. For such purpose we use "rain cloud" plots, which display the
# group-wise empirical distribution along with the corresponding boxplot. They
# are useful when comparing the interactions between categorical and numerical
# variables.

# %%
import ptitprince as pt
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 5))

pt.RainCloud(
    x="sex",
    y="hours-per-week",
    data=adult_census,
    bw=0.2,
    width_viol=0.6,
    ax=ax,
    orient="h",
    move=0.3,
    dodge=True,
)
_ = plt.title("Empirical distribution of \n'hours-per-week' segmented by 'sex'")

# %% [markdown]
# In this case, the boxplot can be misleading. It would simply show that
# 'female' samples work less hours-per-week than 'male' samples on average, but
# looking at the empirical distributions one can notice that the **mode** (the
# value that appears most often) is the same for both values of the variable
# `'sex'` i.e., around 40 hours-per-week. A small under-representation of the
# bumps at 50 and 60 hours-per-week for the 'female' distribution, as well as an
# excess of outliers in the 'male' distribution is what raises the mean of the
# latter.
#
# This is still not enough information to know if the under-respresentation is
# due to the data-collection mechanism or the data-generating process. Another
# thing we can do is to differentiate the distributions above by class, i.e. by
# whether the person earns above 50K or not.

# %%
fig, ax = plt.subplots(figsize=(7, 5))

pt.RainCloud(
    x="sex",
    y="hours-per-week",
    hue=target_name,
    data=adult_census,
    bw=0.2,
    width_viol=0.6,
    ax=ax,
    orient="h",
    alpha=0.65,
    move=0.3,
    dodge=True,
)
_ = plt.title(
    "Empirical distribution of the target \nsegmented by 'hours-per-week' and 'sex'"
)

# %% [markdown]
# Now the peaks at 50 and 60 hours-per-week for the 'female' distribution become
# clearer for the class `">50K"`, while the smaller bump at 20 hours-per-week
# flattens. This hints that the underlying data generation mechanisms might be
# the same regardless of the variable `'sex'`. At least respect to working
# hours.
#
# Similarly, we can visualize the interactions between the target class, the
# categorical variable `'sex'` and the other numerical variables:

# %%
fig, ax = plt.subplots(figsize=(7, 5))

pt.RainCloud(
    x="sex",
    y="education-num",
    hue=target_name,
    data=adult_census,
    bw=0.2,
    width_viol=0.6,
    ax=ax,
    orient="h",
    alpha=0.65,
    move=0.3,
    dodge=True,
)
_ = plt.title(
    "Empirical distribution of the target \nsegmented by 'education-num' and 'sex'"
)

# %% [markdown]
# In this dataset, the education empirical distribution is also similar for both
# values contained in the variable `'sex'`.

# %%
fig, ax = plt.subplots(figsize=(7, 5))

pt.RainCloud(
    x="sex",
    y="age",
    hue=target_name,
    data=adult_census,
    bw=0.2,
    width_viol=0.6,
    ax=ax,
    orient="h",
    alpha=0.65,
    dodge=True,
)
_ = plt.title("Empirical distribution of the target \nsegmented by 'age' and 'sex'")

# %% [markdown]
# In the previous plot we see that the `"<=50K"` class is skewed to low ages for
# the female samples, but once again, the class `">50K"` displays the same
# distribution regardless of the variable `'sex'`.
#
# ## Conclusions and going further
#
# By looking at the previous plots, we confirm that the distributions are
# similar for both 'male' and 'female' categories. There is no _a priori_ reason
# why the selection rate should benefit either group. But in practice, models
# can aquire a bias, for instance, due to the metric minimized during training.
#
# We encourage users interested in [evaluating fairness-related
# metrics](https://fairlearn.org/v0.7.0/quickstart.html#evaluating-fairness-related-metrics)
# and solutions for [mitigating
# disparity](https://fairlearn.org/v0.7.0/quickstart.html#mitigating-disparity)
# to visit [fairlearn.org](https://fairlearn.org).
