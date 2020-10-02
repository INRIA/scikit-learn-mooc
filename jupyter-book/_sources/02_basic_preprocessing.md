---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.6.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# First model with scikit-learn

## Basic preprocessing and model fitting

In this notebook, we present how to build predictive models on tabular
datasets, with numerical features. Categorical features will be discussed
in the next notebook.

In particular we will highlight:
* working with numerical features
* the importance of scaling numerical variables
* evaluate the performance of a model via cross-validation

## Loading the dataset

We will use the same dataset "adult_census" described in the previous notebook.
For more details about the dataset see http://www.openml.org/d/1590.

```{code-cell}
import pandas as pd

df = pd.read_csv("../datasets/adult-census.csv")
```

Let's have a look at the first records of this data frame:

```{code-cell}
df.head()
```

```{code-cell}
target_name = "class"
target = df[target_name].to_numpy()
target
```

For simplicity, we will ignore the "fnlwgt" (final weight) column that was
crafted by the creators of the dataset when sampling the dataset to be
representative of the full census database.

```{code-cell}
data = df.drop(columns=[target_name, "fnlwgt"])
data.head()
```

## Working with numerical data

Numerical data is the most natural type of data used in machine learning
and can (almost) directly be fed to predictive models. We can quickly have a
look at such data by selecting the subset of numerical columns from the
original data.

We will use this subset of data to fit a linear classification model to
predict the income class.

```{code-cell}
data.columns
```

```{code-cell}
data.dtypes
```

```{code-cell}
from sklearn.compose import make_column_selector as selector

numerical_columns_selector = selector(dtype_include=["int", "float"])
numerical_columns = numerical_columns_selector(data)
numerical_columns
```

```{code-cell}
data_numeric = data[numerical_columns]
data_numeric.head()
```

When building a machine learning model, it is important to leave out a
subset of the data which we can use later to evaluate the trained model.
The data used to fit a model is called training data while the one used to
assess a model is called testing data.

Scikit-learn provides a helper function `train_test_split` which will
split the dataset into a training and a testing set. It will also ensure that
the data are shuffled randomly before splitting the data.

```{code-cell}
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data_numeric, target, random_state=42)
```

```{code-cell}
print(
    f"The training dataset contains {data_train.shape[0]} samples and "
    f"{data_train.shape[1]} features")
```

```{code-cell}
print(
    f"The testing dataset contains {data_test.shape[0]} samples and "
    f"{data_test.shape[1]} features")
```

We will build a linear classification model called "Logistic Regression". The
`fit` method is called to train the model from the input (features) and
target data. Only the training data should be given for this purpose.

In addition, check the time required to train the model and the number of
iterations done by the solver to find a solution.

```{code-cell}
from sklearn.linear_model import LogisticRegression
import time

model = LogisticRegression()
start = time.time()
model.fit(data_train, target_train)
elapsed_time = time.time() - start

print(f"The model {model.__class__.__name__} was trained in "
      f"{elapsed_time:.3f} seconds for {model.n_iter_} iterations")
```

Let's ignore the convergence warning for now and instead let's try
to use our model to make some predictions on the first five records
of the held out test set:

```{code-cell}
target_predicted = model.predict(data_test)
target_predicted[:5]
```

```{code-cell}
target_test[:5]
```

```{code-cell}
predictions = data_test.copy()
predictions['predicted-class'] = target_predicted
predictions['expected-class'] = target_test
predictions['correct'] = target_predicted == target_test
predictions.head()
```

To quantitatively evaluate our model, we can use the method `score`. It will
compute the classification accuracy when dealing with a classification
problem.

```{code-cell}
print(f"The test accuracy using a {model.__class__.__name__} is "
      f"{model.score(data_test, target_test):.3f}")
```

This is mathematically equivalent as computing the average number of time
the model makes a correct prediction on the test set:

```{code-cell}
(target_test == target_predicted).mean()
```

## Exercise 1

- What would be the score of a model that always predicts `' >50K'`?
- What would be the score of a model that always predicts `' <= 50K'`?
- Is 81% or 82% accuracy a good score for this problem?

Hint: You can use a `DummyClassifier` and do a train-test split to evaluate
its accuracy on the test set. This
[link](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators)
shows a few examples of how to evaluate the performance of these baseline
models.

Open the dedicated notebook in Jupyter to do this exercise.

+++

Let's now consider the `ConvergenceWarning` message that was raised previously
when calling the `fit` method to train our model. This warning informs us that
our model stopped learning because it reached the maximum number of
iterations allowed by the user. This could potentially be detrimental for the
model accuracy. We can follow the (bad) advice given in the warning message
and increase the maximum number of iterations allowed.

```{code-cell}
model = LogisticRegression(max_iter=50000)
start = time.time()
model.fit(data_train, target_train)
elapsed_time = time.time() - start
```

```{code-cell}
print(
    f"The accuracy using a {model.__class__.__name__} is "
    f"{model.score(data_test, target_test):.3f} with a fitting time of "
    f"{elapsed_time:.3f} seconds in {model.n_iter_} iterations")
```

We now observe a longer training time but no significant improvement in
the predictive performance. Instead of increasing the number of iterations, we
can try to help fit the model faster by scaling the data first. A range of
preprocessing algorithms in scikit-learn allows us to transform the input data
before training a model. 

In our case, we will standardize the data and then train a new logistic
regression model on that new version of the dataset.

```{code-cell}
data_train.describe()
```

```{code-cell}
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_train_scaled = scaler.fit_transform(data_train)
data_train_scaled
```

```{code-cell}
data_train_scaled = pd.DataFrame(data_train_scaled,
                                 columns=data_train.columns)
data_train_scaled.describe()
```

We can easily combine these sequential operations
with a scikit-learn `Pipeline`, which chains together operations and can be
used like any other classifier or regressor. The helper function
`make_pipeline` will create a `Pipeline` by giving as arguments the successive
transformations to perform followed by the classifier or regressor model.

```{code-cell}
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(),
                      LogisticRegression())
start = time.time()
model.fit(data_train, target_train)
elapsed_time = time.time() - start
```

```{code-cell}
print(
    f"The accuracy using a {model.__class__.__name__} is "
    f"{model.score(data_test, target_test):.3f} with a fitting time of "
    f"{elapsed_time:.3f} seconds in {model[-1].n_iter_} iterations")
```

We can see that the training time and the number of iterations is much shorter
while the predictive performance (accuracy) stays the same.

In the previous example, we split the original data into a training set and a
testing set. This strategy has several issues: in the setting where the amount
of data is limited, the subset of data used to train or test will be small;
and the splitting was done in a random manner and we have no information
regarding the confidence of the results obtained.

Instead, we can use cross-validation. Cross-validation consists of
repeating this random splitting into training and testing sets and aggregating
the model performance. By repeating the experiment, one can get an estimate of
the variability of the model performance.

The function `cross_val_score` allows for such experimental protocol by giving
the model, the data and the target. Since there exists several
cross-validation strategies, `cross_val_score` takes a parameter `cv` which
defines the splitting strategy.

```{code-cell}
%%time
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, data_numeric, target, cv=5)
```

```{code-cell}
scores
```

```{code-cell}
print(f"The mean cross-validation accuracy is: "
      f"{scores.mean():.3f} +/- {scores.std():.3f}")
```

Note that by computing the standard-deviation of the cross-validation scores
we can get an idea of the uncertainty of our estimation of the predictive
performance of the model: in the above results, only the first 2 decimals seem
to be trustworthy. Using a single train / test split would not allow us to
know anything about the level of uncertainty of the accuracy of the model.

Setting `cv=5` created 5 distinct splits to get 5 variations for the training
and testing sets. Each training set is used to fit one model which is then
scored on the matching test set. This strategy is called K-fold
cross-validation where `K` corresponds to the number of splits.

The figure helps visualize how the dataset is partitioned
into train and test samples at each iteration of the cross-validation
procedure:

![Cross-validation diagram](../figures/cross_validation_diagram.png)

For each cross-validation split, the procedure trains a model on the
concatenation of the red samples and evaluate the score of the model
by using the blue samples. Cross-validation is therefore computationally
intensive because it requires training several models instead of one.

Note that the `cross_val_score` method above discards the 5 models that
were trained on the different overlapping subset of the dataset.
The goal of cross-validation is not to train a model, but rather to
estimate approximately the generalization performance of a model that
would have been trained to the full training set, along with an estimate
of the variability (uncertainty on the generalization accuracy).

+++

In this notebook we have:
* **split** our dataset into a training dataset and a testing dataset
* fitted a **logistic regression** model
* seen the importance of **scaling numerical variables**
* used the **pipeline** method to fit both the scaler and the logistic regression
* assessed the performance of our model via **cross-validation**
