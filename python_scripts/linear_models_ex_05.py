# %% [markdown]
# # 📝 Exercise 05
# In the previous notebook, we presented a non-penalized logistic regression
# classifier. This classifier accepts a parameter `penalty` to add a
# regularization. The regularization strength is set using the parameter `C`.
#
# In this exercise, we ask you to train a l2-penalized logistic regression
# classifier and to find by yourself the effect of the parameter `C`.
#
# We will start by loading the dataset and create the helper function to show
# the decision separation as in the previous code

# %%
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("../datasets/penguins_classification.csv")
# only keep the Adelie and Chinstrap classes
data = data.set_index("Species").loc[["Adelie", "Chinstrap"]].reset_index()

culmen_columns = ["Culmen Length (mm)", "Culmen Depth (mm)"]
target_column = "Species"
X, y = data[culmen_columns], data[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=0,
)
range_features = {
    feature_name: (X[feature_name].min() - 1, X[feature_name].max() + 1)
    for feature_name in X
}

# %%
import numpy as np
import matplotlib.pyplot as plt


def plot_decision_function(fitted_classifier, range_features, ax=None):
    """Plot the boundary of the decision function of a classifier."""
    from sklearn.preprocessing import LabelEncoder

    feature_names = list(range_features.keys())
    # create a grid to evaluate all possible samples
    plot_step = 0.02
    xx, yy = np.meshgrid(
        np.arange(*range_features[feature_names[0]], plot_step),
        np.arange(*range_features[feature_names[1]], plot_step),
    )

    # compute the associated prediction
    Z = fitted_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = LabelEncoder().fit_transform(Z)
    Z = Z.reshape(xx.shape)

    # make the plot of the boundary and the data samples
    if ax is None:
        _, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.4, cmap="RdBu")
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])

    return ax


# %% [markdown]
# Given the following candidate for the parameter `C`, find out what is the
# effect of the value of this parameter on the decision boundary and on the
# weights magnitude.

# %%
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

Cs = [0.01, 0.1, 1, 10]
logistic_regression = make_pipeline(
    StandardScaler(), LogisticRegression(penalty="l2"))

# %%
# Write your code here.
