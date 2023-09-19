# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # 📃 Solution for Exercise M4.03
#
# The scikit-learn implementation of logistic regression has a `penalty`
# hyperparameter that controls the **type** of regularization to apply, whereas
# the regularization **strength** is set using the `C` hyperparameter.
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

penguins_train, penguins_test = train_test_split(
    penguins, random_state=0, test_size=0.4
)

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
# ## Influence of the parameter `C` on the decision boundary
#
# Given the following candidates for the `C` parameter, find out the impact of
# `C` on the classifier decision boundary. You can use
# `sklearn.inspection.DecisionBoundaryDisplay.from_estimator` to plot the
# decision function boundary.
#
# - How does the value of `C` impact the number of misclassified samples in the
# data set?
# - How does it impact the darkness of the color associated to the decision
# boundary?
# - What does it mean with respect to the confidence of the classifier in
# different regions of the feature space?
# - Does the direction of the decision boundary change when changing the value
# of `C`?

# %%
Cs = [1e-6, 0.01, 0.1, 1, 10, 100, 1e6]

# solution
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import DecisionBoundaryDisplay
import warnings


warnings.filterwarnings("ignore", module="seaborn")

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
        palette=["tab:blue", "tab:red"],
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
# ## Impact of the regularization on the weights
#
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
# For even stronger penalty strengths (e.g. `C = 1e-6`), the weights of both
# features are almost zero. It explains why the decision separation in the plot
# is almost constant in the feature space: the predicted probability is only
# based on the intercept of the model.

# %% [markdown]
# ## Impact of the regularization on with non-linear feature engineering
#
# Repeat the experiment using a non-linear feature engineering
# pipeline, by inserting `Nystroem(kernel="rbf", gamma=1, n_components=100)`
# between the `StandardScaler` and the `LogisticRegression` steps.
#
# - Does the value of `C` still impact the position of the decision boundary
#   and the confidence of the model?
# - What can you say about the impact of `C` on the under-fitting vs
#   over-fitting trade-off?

# %%
from sklearn.kernel_approximation import Nystroem

# solution
classifier = make_pipeline(
    StandardScaler(),
    Nystroem(kernel="rbf", gamma=1.0, n_components=100, random_state=0),
    LogisticRegression(penalty="l2", max_iter=1000),
)

for C in Cs:
    classifier.set_params(logisticregression__C=C)
    classifier.fit(data_train, target_train)
    accuracy = classifier.score(data_test, target_test)

    disp = DecisionBoundaryDisplay.from_estimator(
        classifier,
        data_train,
        response_method="predict_proba",
        plot_method="pcolormesh",
        cmap="RdBu_r",
        alpha=0.8,
        vmin=0.0,
        vmax=1.0,
    )
    DecisionBoundaryDisplay.from_estimator(
        classifier,
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
        palette=["tab:blue", "tab:red"],
        ax=disp.ax_,
    )
    plt.legend(bbox_to_anchor=(1.05, 0.8), loc="upper left")
    plt.title(f"C: {C} \n Accuracy on the test set: {accuracy:.2f}")

# %% [markdown] tags=["solution"]
#
# - For the lowest values of `C`, the overall pipeline is underfits: it
#   constantly predicts the same class probability everywhere, and therefore
#   the majority class, as previously.
# - When `C` increases, the models starts to predict some datapoints from the
#   "Chinstrap" class but the model is not very confident anywhere in the
#   feature space.
# - The decision boundary is no longer a straight line: the linear model is now
#   classifying in the 100-dimensional feature space created by the `Nystroem`
#   transformer. As are result, the decision boundary induced by the overall
#   pipeline is now expressive enough to wrap around the minority class.
# - For `C = 1` in particular, it finds a smooth red blob around most of the
#   "Chinstrap" data points. When moving away from the data points, the model
#   is less confident in its predictions and again tends to predict the
#   majority according to the relative proportion of each class in the training
#   set.
# - For higher values of `C`, the model starts to overfit: it is very confident
#   in its predictions almost everywhere, but it should not be trusted: the
#   model also makes a larger number of mistakes on the test set (not
#   represented here) while adopting a very curvy decision boundary to attempt
#   to fit all the training points, including the noisy ones at the frontier
#   between the two classes. This makes the decision boundary very sensitive to
#   the sampling of the training set and as a result, it does not generalize
#   well in that region. This is confirmed by the lower accuracy on the test
#   set.
#
# Finally, we can also note that the linear model on the raw features was as
# good or better than the best model using non-linear feature engineering. So
# in this case, we did not really need this extra complexity in our pipeline.
#
# So to conclude, when using non-linear feature engineering, it is often
# possible to make the pipeline overfit, even if the original feature space
# is low-dimensional. As a result, it is important to tune the regularization
# parameter in conjunction with the parameters of the transformers (e.g. tuning
# `gamma` would be important here).
