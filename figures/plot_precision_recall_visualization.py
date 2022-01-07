import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.datasets import make_blobs
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

FIGURES_FOLDER = Path(__file__).parent
plt.style.use(FIGURES_FOLDER / "../python_scripts/matplotlibrc")

# Create the dataset for a binary classification problem

feature_columns = ["Feature #0", "Feature #1"]
target_column = "Class"

X, y = make_blobs(
    n_samples=100, centers=[[0, 0], [-1, -1]], random_state=0, cluster_std=0.8)

data_clf = np.concatenate([X, y[:, np.newaxis]], axis=1)
data_clf = pd.DataFrame(
    data_clf, columns=feature_columns + [target_column])
data_clf[target_column] = data_clf[target_column].astype(np.int32)

# Use a simple train-test split

data_clf_train, data_clf_test = train_test_split(data_clf, random_state=0)

data_train = data_clf_train[feature_columns]
data_test = data_clf_test[feature_columns]

target_train = data_clf_train[target_column]
target_test = data_clf_test[target_column]

range_features = {
    feature_name: (data_clf[feature_name].min() - 1,
                   data_clf[feature_name].max() + 1)
    for feature_name in feature_columns
}

# Visualize full dataset

sns.scatterplot(data=data_clf, x=feature_columns[0], y=feature_columns[1],
                hue=target_column, palette=["tab:red", "tab:blue"])
_ = plt.title("Synthetic dataset")

# Create linear model

logistic_regression = make_pipeline(
    StandardScaler(), LogisticRegression(penalty="l2"))

# Define function to visualize the boundary of the decision function

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
    grid = pd.DataFrame(
        np.c_[xx.ravel(), yy.ravel()],
        columns=[feature_names[0], feature_names[1]],
    )

    # compute the associated prediction
    Z = fitted_classifier.predict(grid)
    Z = LabelEncoder().fit_transform(Z)
    Z = Z.reshape(xx.shape)

    # make the plot of the boundary and the data samples
    if ax is None:
        _, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.4, cmap="RdBu_r")

    return ax


# Make the plots and save them
Cs = [0.003, 1]

for C in Cs:
    logistic_regression.set_params(logisticregression__C=C)
    logistic_regression.fit(data_train, target_train)
    target_predicted = logistic_regression.predict(data_test)
    precision = precision_score(target_test, target_predicted, pos_label=1)
    recall = recall_score(target_test, target_predicted, pos_label=1)
    print(f"C={C}, precision={precision:.3f}, recall={recall:.3f}")

    plt.figure()
    ax = sns.scatterplot(
        data=data_clf_test, x=feature_columns[0], y=feature_columns[1],
        hue=target_column, palette=["tab:blue", "tab:red"])
    plot_decision_function(logistic_regression, range_features, ax=ax)
    msg = "Weaker regularization" if C == 1 else "Stronger regularization"
    plt.title(msg + f" (C={C})")
    plt.tight_layout()
    plt.savefig("evaluation_quiz_precision_recall_C" + str(C) + ".svg", facecolor="none", edgecolor="none")
