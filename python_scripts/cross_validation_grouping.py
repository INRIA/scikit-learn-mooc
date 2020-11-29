# %% [markdown]
# # Sample grouping
# We are going to linger into the concept of samples group. As in the previous
# section, we will give an example to highlight some surprising results. This
# time, we will use the handwritten digits dataset.

# %%
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(), LogisticRegression())

# %% [markdown]
# We will use the same baseline model. We will use a `KFold` cross-validation
# without shuffling the data at first.

# %%
from sklearn.model_selection import cross_validate, KFold

cv = KFold(shuffle=False)
results = cross_validate(model, X, y, cv=cv, n_jobs=-1)
test_score_no_shuffling = results["test_score"]
print(f"The average accuracy is {test_score_no_shuffling.mean():.3f}")

# %%
cv = KFold(shuffle=True)
results = cross_validate(model, X, y, cv=cv, n_jobs=-1)
test_score_with_shuffling = results["test_score"]
print(f"The average accuracy is {test_score_with_shuffling.mean():.3f}")

# %% [markdown]
# We observe that shuffling the data allows to improving the mean accuracy.
# We could go a little further and plot the distribution of the generalization
# score.

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk")

sns.displot(data=pd.DataFrame(
    [test_score_no_shuffling, test_score_with_shuffling],
    index=["KFold without shuffling", "KFold with shuffling"],
).T, kde=True, bins=10)
plt.xlim([0.8, 1.0])
plt.xlabel("Accuracy score")

# %% [markdown]
# The cross-validation generalization error that uses the shuffling has less
# variance than the one that does not impose any shuffling. It means that some
# specific fold leads to a low score in the unshuffle case.

# %%
print(test_score_no_shuffling)

# %% [markdown]
# Thus, there is an underlying structure in the data that shuffling will break
# and get better results. To get a better understanding, we should read the
# documentation shipped with the dataset.

# %%
print(digits.DESCR)

# %% [markdown]
# If we read carefully, 13 writers wrote the digits of our dataset, accounting
# for a total amount of 1797 samples. Thus, a writer wrote several times the
# same numbers. Let's suppose that the writer samples are grouped.
# Subsequently, not shuffling the data will keep all writer samples together
# either in the training or the testing sets. Mixing the data will break this
# structure, and therefore digits written by the same writer will be available
# in both the training and testing sets.
#
# Besides, a writer will usually tend to write digits in the same manner. Thus,
# our model will learn to identify a writer's pattern for each digit instead of
# recognizing the digit itself.
#
# We can solve this problem by ensuring that the data associated with a writer
# should either belong to the training or the testing set. Thus, we want to
# group samples for each writer.
#
# Here, we will manually define the group for the 13 writers.

# %%
from itertools import count
import numpy as np

# defines the lower and upper bounds of sample indices
# for each writer
writer_boundaries = [
    0, 130, 256, 386, 516, 646, 776, 915, 1029,
    1157, 1287, 1415, 1545, 1667, 1797
]
groups = np.zeros_like(y)

for group_id, lower_bound, upper_bound in zip(
    count(),
    writer_boundaries[:-1],
    writer_boundaries[1:]
):
    groups[lower_bound:upper_bound] = group_id
groups

# %% [markdown]
# Once we group the digits by writer, we can use cross-validation to take this
# information into account: the class containing `Group` should be used.

# %%
from sklearn.model_selection import GroupKFold

cv = GroupKFold()
results = cross_validate(model, X, y, groups=groups, cv=cv, n_jobs=-1)
test_score = results["test_score"]
print(f"The average accuracy is {test_score.mean():.3f}")

# %% [markdown]
# We see that this strategy is less optimistic regarding the model performance.
# However, this is the most reliable if our goal is to make handwritten digits
# recognition writers independent.
