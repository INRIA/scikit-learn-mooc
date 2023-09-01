# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # 📃 Solution for Exercise M4.03
#
# The parameter `penalty` can control the **type** of regularization to use,
# whereas the regularization **strength** is set using the parameter `C`.
# Setting`penalty="none"` is equivalent to an infinitely large value of `C`. In
# this exercise, we ask you to train a logistic regression classifier using the
# `penalty="l2"` regularization (which happens to be the default in
# scikit-learn) to find by yourself the effect of the parameter `C`.
#
# We start by loading the dataset.

# %% [markdown]
# ```{note}
# If you want a deeper overview regarding this dataset, you can refer to the
# Appendix - Datasets description section at the end of this MOOC.
# ```

# %%
import pandas as pd

penguins = pd.read_csv("../datasets/penguins_classification.csv")
# only keep the Adelie and Chinstrap classes
penguins = (
    penguins.set_index("Species").loc[["Adelie", "Chinstrap"]].reset_index()
)

culmen_columns = ["Culmen Length (mm)", "Culmen Depth (mm)"]
target_column = "Species"

# %%
from sklearn.model_selection import train_test_split

penguins_train, penguins_test = train_test_split(penguins, random_state=0)

data_train = penguins_train[culmen_columns]
data_test = penguins_test[culmen_columns]

target_train = penguins_train[target_column]
target_test = penguins_test[target_column]

# %% [markdown]
# First, let's create our predictive model.

# %%
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

logistic_regression = make_pipeline(
    StandardScaler(), LogisticRegression(penalty="l2")
)

# %% [markdown]
# Given the following candidates for the `C` parameter, find out the impact of
# `C` on the classifier decision boundary. You can use
# `sklearn.inspection.DecisionBoundaryDisplay.from_estimator` to plot the
# decision function boundary.

# %%
Cs = [0.0001, 0.01, 0.1, 1, 1000]

# solution
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import DecisionBoundaryDisplay

for C in Cs:
    logistic_regression.set_params(logisticregression__C=C)
    logistic_regression.fit(data_train, target_train)
    accuracy = logistic_regression.score(data_test, target_test)

    disp = DecisionBoundaryDisplay.from_estimator(
        logistic_regression,
        data_train,
        response_method="predict_proba",
        plot_method="pcolormesh",
        cmap="RdBu_r",
        alpha=0.8,
        # Setting vmin and vmax to the extreme values of the probability in
        # order to make sure that 0.5 is mapped to white (the middle) of the
        # blue-red colormap.
        vmin=0.0,
        vmax=1.0,
    )
    DecisionBoundaryDisplay.from_estimator(
        logistic_regression,
        data_train,
        response_method="predict_proba",
        plot_method="contour",
        linestyles="--",
        linewidths=1,
        alpha=0.8,
        levels=[0.5],  # 0.5 probability contour line
        ax=disp.ax_,
    )
    sns.scatterplot(
        data=penguins_train,
        x=culmen_columns[0],
        y=culmen_columns[1],
        hue=target_column,
        palette=["tab:red", "tab:blue"],
        ax=disp.ax_,
    )
    plt.legend(bbox_to_anchor=(1.05, 0.8), loc="upper left")
    plt.title(f"C: {C} \n Accuracy on the test set: {accuracy:.2f}")

# %% [markdown] tags=["solution"]
#
# On this series of plots we can observe several important points:
#
# - the darker the color, the more confident the classifier is in its
#   predictions. Indeed, the darker the color, the closer the predicted
#   probability is to 0 or 1;
# - for lower values of `C` (stronger regularization), the classifier is less
#   confident in its predictions (the near-white area is larger);
# - for higher values of `C` (weaker regularization), the classifier is more
#   confident: the areas with dark blue (very confident in predicting Adelie)
#   and dark red (very confident in predicting Chinstrap) nearly cover the
#   entire feature space;
# - the direction of the straight line separating the two classes is impacted
#   by the choice of `C`: the smaller `C` (the higher the regularization), the
#   more the decision bounday is influenced almost uniformly by all the
#   datapoints: the decision boundary is almost perpendicular to the "Culmen
#   Length (mm)" feature.
# - the higher the value of `C` (the weaker the regularization), the more the
#   decision boundary is influenced by few traing points very close to the
#   decision boundary, in particular by the misclassified points caused by the
#   presence of noise in the data, making this classification task non-linearly
#   separable.
# - also note that for small values of `C`, the decision boundary is almost
#   vertical: the model is almost only using the feature named "Culmen Length
#   (mm)" to make its predictions. We will explain this behavior in the next
#   part of the exercise.
# - finally, the 2 classes are imbalanced: there are approximately two times
#   more Adelie than Chinstrap penguins. This explains why the decision
#   boundary is shifted to the right when `C` gets smaller. Indeed, when `C` is
#   near zero, the model is nearly always predicting the same class probability
#   almost everywhere in the feature space. This class is the one that matches
#   the proportion of each class in the training set. In our case, there are
#   more Adelie than Chinstrap penguins in the training set: as a results the
#   most regularized model predicts light blue almost everywhere in the feature
#   space.

# %% [markdown]
# Look at the impact of the `C` hyperparameter on the magnitude of the weights.

# %%
# solution
lr_weights = []
for C in Cs:
    logistic_regression.set_params(logisticregression__C=C)
    logistic_regression.fit(data_train, target_train)
    coefs = logistic_regression[-1].coef_.squeeze()
    lr_weights.append(pd.Series(coefs, index=culmen_columns))


# %% tags=["solution"]
lr_weights = pd.concat(lr_weights, axis=1, keys=[f"C: {C}" for C in Cs])
lr_weights.plot.barh()
_ = plt.title("LogisticRegression weights depending of C")

# %% [markdown] tags=["solution"]
#
# We see that a small `C` will shrink the weights values toward zero. It means
# that a small `C` provides a more regularized model. Thus, `C` behaves as the
# inverse of the `alpha` parameter in the `Ridge` model.
#
# Besides, with a stronger penalty (e.g. `C = 0.01`), the weight of the feature
# named "Culmen Depth (mm)" is almost zero. It explains why the decision
# separation in the plot is almost perpendicular to the "Culmen Length (mm)"
# feature.
#
# For even stronger penalty strengths (e.g. `C = 0.0001`), the weights of both
# features are almost zero. It explains why the decision separation in the plot
# is almost constant in the feature space: the predicted probability is only
# based on the intercept of the model.
