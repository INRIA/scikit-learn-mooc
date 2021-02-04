# %% [markdown]
# # 📝 Exercise 02
#
# The aim of this exercise it to explore some attributes available in
# scikit-learn random forest.
#
# First, we will fit the penguins regression dataset.

# %%
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("../datasets/penguins_regression.csv")
feature_names = ["Flipper Length (mm)"]
target_name = "Body Mass (g)"
X, y = data[feature_names], data[target_name]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# %% [markdown]
# Create a random forest containing three trees. Train the forest and
# check the performance on the testing set.

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

# ```{tip}
# The trees contained in the forest that you created can be accessed
# with the attribute `estimators_`.
# ```

# %%
# Write your code here.
