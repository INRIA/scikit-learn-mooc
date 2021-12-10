# -*- coding: utf-8 -*-
# %% [markdown]
# # 📝 Exercise M6.02
#
# The aim of this exercise it to explore some attributes available in
# scikit-learn's random forest.
#
# First, we will fit the penguins regression dataset.

# %%
import pandas as pd
from sklearn.model_selection import train_test_split

penguins = pd.read_csv("../datasets/penguins_regression.csv")
feature_name = "Flipper Length (mm)"
target_name = "Body Mass (g)"
data, target = penguins[[feature_name]], penguins[target_name]
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=0)

# %% [markdown]
# ```{note}
# If you want a deeper overview regarding this dataset, you can refer to the
# Appendix - Datasets description section at the end of this MOOC.
# ```

# %% [markdown]
# Create a random forest containing three trees. Train the forest and
# check the generalization performance on the testing set in terms of mean
# absolute error.

# %%
# Write your code here.

# %% [markdown]
# The next steps of this exercise are to:
#
# - create a new dataset containing the penguins with a flipper length
#   between 170 mm and 230 mm;
# - plot the training data using a scatter plot;
# - plot the decision of each individual tree by predicting on the newly
#   created dataset;
# - plot the decision of the random forest using this newly created dataset.
#
# ```{tip}
# The trees contained in the forest that you created can be accessed
# with the attribute `estimators_`.
# ```

# %%
# Write your code here.
