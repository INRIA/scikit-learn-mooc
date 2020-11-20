# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Evaluation of your predictive model: which metric to choose?

# %% [markdown]
# ## Introduction
# Machine-learning models rely on optimizing an objective function, by seeking
# its minimum or maximum. It is important to understand that this objective
# function is usually decoupled from the evaluation metric that we want to
# optimize in practice. The objective function serves as a proxy for the
# evaluation metric.
# FIXME: add information about a loss function depending of the notebooks
# presented before the notebook about metrics.
#
# While other notebooks will give insight about machine-learning algorithms and
# their associated objective functions, in this notebook we will focus on the
# metrics used to evaluate the performance of a predictive model.
#
# Evaluation metric selection will mainly depend on the model chosen to
# solve our data science problem.

# %% [markdown]
# ## Classification
# We can recall that in a classification setting, the target `y` is categorical
# rather than continuous. We will use the blood transfusion dataset that will
# be fetched from OpenML.

# %%
import pandas as pd

data = pd.read_csv("../datasets/blood_transfusion.csv")
X, y = data[["V1", "V2", "V3", "V4"]], data["Class"]
# Make columns and classes more human-readable
X.columns = ["Recency", "Frequency", "Monetary", "Time"]
y = y.apply(
    lambda x: "donated" if x == 2 else "not donated"
).astype("category")
y.cat.categories

# %% [markdown]
# We can see that the target `y` contains 2 categories corresponding to whether
# or not a subject gave blood or not. We will use a logistic regression
# classifier to predict this outcome.
#
# First, we split the data into a training and a testing set.

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=True, random_state=0, test_size=0.5
)

# %% [markdown]
# Once our data are split, we can learn a logistic regression classifier using
# only the training data, keeping the testing data for evaluation of the
# model.

# %%
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# %% [markdown]
# Now that our classifier is trained, we can provide some information about a
# subject and the classifier can predict whether or not the subject will donate
# blood.
#
# Let's create a synthetic sample for a new potential
# donor: he/she donated blood 6 months ago and has given a total of 1000 c.c.
# of blood, twice in the past. He/she gave blood for the first time 20
# months ago.

# %%
new_donor = [[6, 2, 1000, 20]]
classifier.predict(new_donor)

# %% [markdown]
# With this information, our classifier predicts that this synthetic subject
# is more likely to not donate blood again. However, we cannot check if the
# prediction is correct or not (we do not know the true target value). That's
# the purpose of the testing set. First, we predict whether or not a
# subject will give blood with the help of the trained classifier.

# %%
y_pred = classifier.predict(X_test)
y_pred[:5]

# %% [markdown]
# ### Accuracy as a baseline
# Now that we have these predictions, we can compare them with the true
# predictions (sometimes called ground-truth) which we did not use up to now.

# %%
y_test == y_pred

# %% [markdown]
# In the comparison above, a `True` value means that the value predicted by our
# classifier is identical to the real `prediction` while a `False` means that
# our classifier made a mistake. One way to get an overall statistic that tells
# us how good the performance of our classifier is, is to compute the number of
# times our classifier was right and divide it by the number of samples in our
# set.

# %%
import numpy as np

np.mean(y_test == y_pred)

# %% [markdown]
# This measure is also known as the accuracy. Here, our classifier is 78%
# accurate at classifying if a subject will give blood. `scikit-learn` provides
# a function that computes this metric in the module `sklearn.metrics`.

# %%
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)

# %% [markdown]
# Scikit-learn also has a method named `score`, built into
# `LogisticRegression`, which computes the accuracy score.

# %%
classifier.score(X_test, y_test)

# %% [markdown]
# ### Confusion matrix and derived metrics
# The comparison that we did above and the accuracy that we calculated did not
# take into account the type of error our classifier was making. Accuracy
# is an aggregate of the errors made by the classifier. We may be interested
# in finer granularity - to know independently what the error is for each of
# the two following cases:
# - we predicted that a person will give blood but she/he did not;
# - we predicted that a person will not give blood but she/he did.

# %%
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(classifier, X_test, y_test)

# %% [markdown]
# The in-diagonal numbers are related to predictions that were correct
# while off-diagonal numbers are related to incorrect predictions
# (misclassifications). We now know the four types of correct and erroneous
# predictions:
#
# * the top left corner are true positives (TP) and corresponds to people
#   who gave blood and was predicted as such by the classifier;
# * the bottom right corner are true negatives (TN) and correspond to
#   a people who did not give blood and was predicted as such by the
#   classifier;
# * the top right corner are false negatives (FN) and correspond to
#   people who gave blood but was predicted to not have given blood;
# * the bottom left corner are false positives (FP) and correspond to
#   people who did not give blood but was predicted to have given blood.
#
# Once we have split this information, we can compute statistics tp
# highlight the performance of our classifier in a particular setting. For
# instance, we could be interested in the fraction of people who really gave
# blood when the classifier predicted so or the fraction of people predicted
# to have given blood out of the total population that actually did so.
#
# The former statistic, known as the precision, is defined as TP / (TP + FP)
# and represents how likely the person actually gave blood when the classifier
# predicted that they did.
# The latter statistic, known as the recall, defined as TP / (TP + FN) and
# assesses how well the classifier is able to correctly identify people who
# did give blood.
# We could, similar to accuracy, manually compute these values
# but scikit-learn provides functions to compute these statistics.

# %%
from sklearn.metrics import precision_score, recall_score

print(
    f"Precision score: {precision_score(y_test, y_pred, pos_label='donated')}"
    f"\nRecall score: {recall_score(y_test, y_pred, pos_label='donated')}"
)

# %% [markdown]
# These results are in line with what was seen in the confusion matrix.
# Looking at the left column, more than half of the "donated" predictions were
# correct, leading
# to a precision above 0.5. However, our classifier mislabeled a lot of people
# who gave blood as "not donated", leading to a very low recall of around 0.1.
#
# ### The issue of class imbalance
# At this stage, we could ask ourself a reasonable question. While the accuracy
# did not look bad (i.e. 77%), the F1 score is relatively low (i.e. 21%).
#
# As we mentioned, precision and recall only focuses on samples predicted to
# be positive, while
# accuracy takes both into account. In addition,
# we did not look at the ratio of classes (labels).
# We could check this ratio in the training set.

# %%
from collections import Counter

class_counts = pd.Series(Counter(y_train))
class_counts /= class_counts.sum()
class_counts

# %% [markdown]
# We can observe that the positive class, `'donated'`, comprises only 24% of
# the of the samples. The good accuracy of our classifier is then linked
# to its ability to predict correctly the negative class `'not donated'`
# which may or may not be relevant, depending on the application. We can
# illustrate the issue using a dummy classifier as a baseline.

# %%
from sklearn.dummy import DummyClassifier

dummy_classifier = DummyClassifier(
    strategy="constant", constant="not donated"
)
dummy_classifier.fit(X_train, y_train).score(X_test, y_test)

# %% [markdown]
# With the dummy classifier, which always predicts the negative class
# `'not donated'`,
# we obtain an accuracy score of 76%. Therefore, it means that this classifier,
# without learning anything from the data `X`, is capable of predicting as
# accurately as our logistic regression model.
#
# The problem illustrated above is also known as the class imbalance problem.
# When the classes are imbalanced, accuracy should not be used. In this case,
# one should either use
# the precision, recall, or F1 score as presented above or the balanced
# accuracy score instead of accuracy.

# %%
from sklearn.metrics import balanced_accuracy_score

balanced_accuracy_score(y_test, y_pred)
# %% [markdown]
# The balanced accuracy is equivalent to accuracy in the context of
# balanced classes. It is defined as the average recall obtained on each class.
#
# ### Evaluation and different probability thresholds
#
# All statistics that we presented up to now rely on `classifier.predict` which
# outputs the most likely label. We haven't made use use of the probability
# associated with this prediction, which gives the confidence of the
# classifier in this prediction. By default, the prediction of a classifier
# corresponds to a threshold of 0.5 probability in a binary classification
# problem. We can quickly check this relationship with the classifier that
# we trained.

# %%
y_proba = pd.DataFrame(
    classifier.predict_proba(X_test),
    columns=classifier.classes_
)
y_proba[:5]

# %%
y_pred = classifier.predict(X_test)
y_pred[:5]

# %%
# Since probabilities sum to 1 we can get the class with the highest
# probability without using the threshold 0.5.
equivalence_pred_proba = (
    y_proba.idxmax(axis=1).to_numpy() == y_pred
)
np.all(equivalence_pred_proba)

# %% [markdown]
# The default decision threshold (0.5) might not be the best threshold that
# leads to optimal performance of our classifier. In this case, one can vary
# the decision threshold, and therefore the underlying prediction, and compute
# the same statistics presented earlier. Usually, the two metrics recall and
# precision are computed and plotted on a graph. Each metric plotted on a
# graph axis and each point on
# the graph corresponds to a specific decision threshold. Let's start by
# computing the precision-recall curve.

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

y_pred = classifier.predict_proba(X_test)
pos_label = "donated"
precision, recall, threshold = precision_recall_curve(
    y_test, y_pred[:, 0], pos_label=pos_label,
)
average_precision = average_precision_score(
    y_test, y_pred[:, 0], pos_label=pos_label,
)
plt.plot(
    recall, precision,
    color="tab:orange", linewidth=3,
    marker=".", markerfacecolor="tab:blue", markeredgecolor="tab:blue",
    label=f"Average Precision: {average_precision:.2f}",
)
plt.xlabel(f"Recall\n (Positive label: {pos_label})")
plt.ylabel(f"Precision\n (Positive label: {pos_label})")
plt.legend()

# # FIXME: to be used when solved in scikit-learn
# from sklearn.metrics import plot_precision_recall_curve

# disp = plot_precision_recall_curve(
#     classifier, X_test, y_test, pos_label='donated',
# )

# %% [markdown]
# On this curve, each blue dot corresponds to a level of probability
# which we used as a decision threshold. We can see that by varying this
# decision threshold, we get different precision vs. recall values.
#
# A perfect classifier would have a precision of 1 for all recall
# values. A metric characterizing the curve is linked to the area under the
# curve (AUC) and is named average precision. With an ideal classifier, the
# average precision would be 1.
#
# The precision and recall metric focuses on the positive class however, one
# might be interested in the compromise between accurately discriminating the
# positive class and accurately discriminating the negative classes. The
# statistics used for this are sensitivity and specificity. Sensitivity is just
# another name for recall. However, specificity measures the proportion of
# correctly classified samples in the negative class defined as:
# TN / (TN + FP). Similar to the precision-recall curve, sensitivity and
# specificity are generally plotted as a curve called the receiver operating
# characteristic (ROC) curve. Below is such a curve:

# %%
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

fpr, tpr, threshold = roc_curve(y_test, y_pred[:, 0], pos_label=pos_label)
# FIXME: roc_auc_score has a bug and we need to give the inverse probability
# vector. Should be changed when the following is merged and released:
# https://github.com/scikit-learn/scikit-learn/pull/17594
roc_auc = roc_auc_score(y_test, y_pred[:, 1])
plt.plot(
    fpr, tpr,
    color="tab:orange", linewidth=3,
    marker=".", markerfacecolor="tab:blue", markeredgecolor="tab:blue",
    label=f"ROC-AUC: {roc_auc:.2f}"
)
plt.plot([0, 1], [0, 1], "--", color="tab:green", label="Chance")
plt.xlabel(f"1 - Specificity\n (Positive label: {pos_label})")
plt.ylabel(f"Sensitivity\n (Positive label: {pos_label})")
plt.legend()

# # FIXME: to be used when solved in scikit-learn
# from sklearn.metrics import plot_roc_curve

# plot_roc_curve(classifier, X_test, y_test, pos_label='donated')

# %% [markdown]
# This curve was built using the same principle as the precision-recall
# curve: we vary the probability threshold for determining "hard" prediction
# and compute the metrics. As with the precision-recall curve, we can
# compute the area under the ROC (ROC-AUC) to characterize the performance of
# our classifier. However, it is important to observer that the lower bound
# of the ROC-AUC is 0.5. Indeed, we show the performance of a dummy
# classifier (the green dashed line) to show that the even worst performance
# obtained will always be above this line.
#
# ### Link between confusion matrix, precision-recall curve and ROC curve
#
# TODO: ipywidgets to play with interactive curve


# %%
def plot_pr_curve(classifier, X_test, y_test, pos_label,
                  probability_threshold, ax):
    y_pred = classifier.predict_proba(X_test)
    precision, recall, threshold = precision_recall_curve(
        y_test, y_pred[:, 0], pos_label=pos_label,
    )
    average_precision = average_precision_score(
        y_test, y_pred[:, 0], pos_label=pos_label,
    )
    ax.plot(
        recall, precision,
        color="tab:orange", linewidth=3,
        label=f"Average Precision: {average_precision:.2f}",
    )
    threshold_idx = np.searchsorted(
        threshold, probability_threshold,
    )
    ax.plot(
        recall[threshold_idx], precision[threshold_idx],
        color="tab:blue", marker=".", markersize=10,
    )
    ax.plot(
        [recall[threshold_idx], recall[threshold_idx]],
        [0, precision[threshold_idx]],
        '--', color="tab:blue",
    )
    ax.plot(
        [0, recall[threshold_idx]],
        [precision[threshold_idx], precision[threshold_idx]],
        '--', color="tab:blue",
    )
    ax.set_xlabel(f"Recall")
    ax.set_ylabel(f"Precision")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend()
    return ax


# %%
def plot_roc_curve(classifier, X_test, y_test, pos_label,
                   probability_threshold, ax):
    y_pred = classifier.predict_proba(X_test)
    fpr, tpr, threshold = roc_curve(y_test, y_pred[:, 0], pos_label=pos_label)
    roc_auc = roc_auc_score(y_test, y_pred[:, 1])
    ax.plot(
        fpr, tpr,
        color="tab:orange", linewidth=3,
        label=f"ROC-AUC: {roc_auc:.2f}"
    )
    ax.plot([0, 1], [0, 1], "--", color="tab:green", label="Chance")
    threshold_idx = np.searchsorted(
        threshold[::-1], probability_threshold,
    )
    threshold_idx = len(threshold) - threshold_idx - 1
    ax.plot(
        fpr[threshold_idx], tpr[threshold_idx],
        color="tab:blue", marker=".", markersize=10,
    )
    ax.plot(
        [fpr[threshold_idx], fpr[threshold_idx]],
        [0, tpr[threshold_idx]],
        '--', color="tab:blue",
    )
    ax.plot(
        [0, fpr[threshold_idx]],
        [tpr[threshold_idx], tpr[threshold_idx]],
        '--', color="tab:blue",
    )
    ax.set_xlabel(f"1 - Specificity")
    ax.set_ylabel(f"Sensitivity")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend()
    return ax


# %%
def plot_confusion_matrix_with_threshold(classifier, X_test, y_test, pos_label,
                                         probability_threshold, ax):
    from itertools import product
    from sklearn.metrics import confusion_matrix

    class_idx = np.where(classifier.classes_ == pos_label)[0][0]
    n_classes = len(classifier.classes_)

    y_pred = classifier.predict_proba(X_test)
    y_pred = (y_pred[:, class_idx] > probability_threshold).astype(int)

    cm = confusion_matrix(
        (y_test == pos_label).astype(int), y_pred,
    )
    im_ = ax.imshow(cm, interpolation='nearest')

    text_ = None
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

    text_ = np.empty_like(cm, dtype=object)

    # print text with appropriate color depending on background
    thresh = (cm.max() + cm.min()) / 2.0

    for i, j in product(range(n_classes), range(n_classes)):
        color = cmap_max if cm[i, j] < thresh else cmap_min

        text_cm = format(cm[i, j], '.2g')
        if cm.dtype.kind != 'f':
            text_d = format(cm[i, j], 'd')
            if len(text_d) < len(text_cm):
                text_cm = text_d

        text_[i, j] = ax.text(
            j, i, text_cm, ha="center", va="center", color=color
        )

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=classifier.classes_[[int(not bool(class_idx)), class_idx]],
        yticklabels=classifier.classes_[[int(not bool(class_idx)), class_idx]],
        ylabel="True label",
        xlabel="Predicted label"
    )


# %%
def plot_pr_roc(threshold):
    # FIXME: we could optimize the plotting by only updating the the
    fig, axs = plt.subplots(ncols=3, figsize=(21, 6))
    plot_pr_curve(
        classifier, X_test, y_test, pos_label="donated",
        probability_threshold=threshold, ax=axs[0],
    )
    plot_roc_curve(
        classifier, X_test, y_test, pos_label="donated",
        probability_threshold=threshold, ax=axs[1]
    )
    plot_confusion_matrix_with_threshold(
        classifier, X_test, y_test, pos_label="donated",
        probability_threshold=threshold, ax=axs[2]
    )
    fig.suptitle("Overall performance with positive class 'donated'")


# %%
def plot_pr_roc_interactive():
    from ipywidgets import interactive, FloatSlider
    slider = FloatSlider(min=0, max=1, step=0.01, value=0.5)
    return interactive(plot_pr_roc, threshold=slider)


# %%
plot_pr_roc_interactive()

# %% [markdown]
# ## Regression
# Unlike in classification problems, the target `y` is a continuous
# variable in regression problems. Therefore, classification metrics cannot
# be used to evaluate the performance of regression models. Instead, there
# exists a set of metrics dedicated to regression.

# %%
data = pd.read_csv(
    ("https://raw.githubusercontent.com/christophM/interpretable-ml-book/"
     "master/data/bike.csv"),
)
# rename the columns with human-readable names
data = data.rename(columns={
    "yr": "year", "mnth": "month", "temp": "temperature", "hum": "humidity",
    "cnt": "count", "days_since_2011": "days since 2011"
})
# convert the categorical columns with a proper category data type
for col in data.columns:
    if data[col].dtype.kind == "O":
        data[col] = data[col].astype("category")

# separate the target from the original data
X = data.drop(columns=["count"])
y = data["count"]

# %%
X.head()

# %%
plt.hist(y, bins=50, density=True)
plt.xlabel("Number of bike rentals")
plt.ylabel("Probability")
plt.title("Target distribution")

# %% [markdown]
# Our problem can be formulated as follows: we would like to infer the number
# of bike rentals in a day using information about the day. The number of bike
# rentals is a number that can vary in the interval [0, max_number_of_bikes).
# As in the previous section, we will train a
# model and evaluate its performance while introducing different
# regression metrics.
#
# First, we split the data into training and a testing sets.

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=True, random_state=0
)

# %% [markdown]
# ### Baseline model
# We will use a random forest as a model. However, we first need to check the
# type of data that we are dealing with:

# %%
X_train.info()

# %% [markdown]
# While some features are numeric, some have been tagged as `category`. These
# features need to be encoded such that our random forest can
# deal with them. The simplest solution is to use an `OrdinalEncoder`.
# Regarding, the numerical features, we don't need to do anything. Thus, we
# will create preprocessing steps to take care of the encoding.

# %%
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OrdinalEncoder

categorical_selector = selector(dtype_include="category")
preprocessor = make_column_transformer(
    (OrdinalEncoder(), categorical_selector),
    remainder="passthrough",
)

X_train_preprocessed = pd.DataFrame(
    preprocessor.fit_transform(X_train),
    columns=(
        categorical_selector(X_train) +
        [col for col in X_train.columns
         if col not in categorical_selector(X_train)]
    )
)
X_train_preprocessed.head()

# %% [markdown]
# Just to have some insight about the preprocessing, we preprocess
# the training data show the result. We can observe that the original strings
# are now encoded with numbers. We can now create our model.

# %%
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor

regressor = make_pipeline(preprocessor, RandomForestRegressor())
regressor.fit(X_train, y_train)

# %% [markdown]
# As for scikit-learn classifiers, scikit-learn regressors have a `score`
# method that computes the
# :math:`R^2` score (also known as the coefficient of determination):

# %%
regressor.score(X_test, y_test)

# %% [markdown]
# The :math:`R^2` score represents the proportion of variance of the target
# that is explained by the independent variables in the model. The best score
# possible
# is 1 but there is no lower bound. However, a model that predicts the
# expected value of the target would get a score of 0.

# %%
from sklearn.dummy import DummyRegressor

dummy_regressor = DummyRegressor(strategy="mean")
dummy_regressor.fit(X_train, y_train).score(X_test, y_test)

# %% [markdown]
# The :math:`R^2` score gives insight into the goodness of fit of the
# model. However, this score cannot be compared from one dataset to another and
# the value obtained does not have a meaningful interpretation relative the
# original unit of the target. If we wanted to get an interpretable score, we
# would be interested in the median or mean absolute error.

# %%
from sklearn.metrics import mean_absolute_error

y_pred = regressor.predict(X_test)
print(
    f"Mean absolute error: {mean_absolute_error(y_test, y_pred):.0f}"
)

# %% [markdown]
# By computing the mean absolute error, we can interpret that our model is
# predicting on average 507 bike rentals away from the truth. A disadvantage
# of this metric is that the mean can be
# impacted by large error. For some applications, we might not want these
# large errors to have such a big influence on our metric. In this case we can
# use the median absolute error.

# %%
from sklearn.metrics import median_absolute_error

print(
    f"Median absolute error: {median_absolute_error(y_test, y_pred):.0f}"
)

# %% [markdown]
# This metric tells us that, our model makes a median error of 405 bikes.
# FIXME: **not sure how to introduce the `mean_squared_error`.**

# %% [markdown]
# In addition of metrics, we can visually represent the results by plotting
# the predicted values versus the true values.


# %%
def plot_predicted_vs_actual(y_true, y_pred, title=None):
    plt.scatter(y_true, y_pred)

    max_value = np.max([y_true.max(), y_pred.max()])
    plt.plot(
        [0, max_value],
        [0, max_value],
        color="tab:orange",
        linewidth=3,
        label="Perfect fit",
    )

    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.axis("square")
    plt.legend()
    if title is not None:
        plt.title(title)


plot_predicted_vs_actual(y_test, y_pred)

# %% [markdown]
# On this plot, correct predictions would lie on the diagonal line. This plot
# allows us to detect if the model makes errors in a consistent way, i.e.
# has some bias.
#
# Let's take an example using the house prices in Ames.

# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

data = pd.read_csv("../datasets/house_prices.csv")
X, y = data.drop(columns="SalePrice"), data["SalePrice"]
X = X.select_dtypes(np.number)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# %% [markdown]
# We will fit a ridge regressor on the data and plot the prediction versus the
# actual values.

# %%
model = make_pipeline(StandardScaler(), RidgeCV())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

plot_predicted_vs_actual(y_test, y_pred, title="House prices in Ames")

# %% [markdown]
# On this plot, we see that for the large True price values, our model tends to
# under-estimate the price of the house. Typically, this issue arises when
# the target to predict does not follow a normal distribution. In these cases
# the model would benefit from target transformation.

# %%
from sklearn.preprocessing import QuantileTransformer
from sklearn.compose import TransformedTargetRegressor

model_transformed_target = TransformedTargetRegressor(
    regressor=model,
    transformer=QuantileTransformer(
        n_quantiles=900, output_distribution="normal"
    ),
)
model_transformed_target.fit(X_train, y_train)
y_pred = model_transformed_target.predict(X_test)

plot_predicted_vs_actual(y_test, y_pred, title="House prices in Ames")

# %% [markdown]
# Thus, once we transformed the target, we see that we corrected some of the
# high values.
#
# ## Summary
# In this notebook, we presented the metrics and plots useful to evaluate and
# get insights about models. We both focus on regression and classification
# problems.

# %%
