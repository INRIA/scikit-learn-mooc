# Glossary

```{warning}
This is work in progress. If you are following the beta, you should probably not
review the glossary.

If you are trying to use the glossary, some external glossaries are provided
at the bottom of this page.

While the work on the glossary is going on, there is a chance that some content
may still be useful.
```

This glossary is supposed to describe the main terms used in this course. For
terms that you can not find here, we added useful glossaries at the bottom of
this page.

## Main terms used in this course

### classification

Type of problems where the goal is to predict a target that can take finite set
of values.

Example of classification problems are:

- predicting the type of Iris (setosa, versicolor, virginica) from their petal
  and sepal measurements
- predicting whether patients has a particular disease from the result of their
  medical tests
- predicting whether an email is a spam or not from the email content, sender,
  title, etc ...

When the predicted label can have two values, it is called binary
classification. This the case for the medical and spam use cases above.

When the predicted label can have at least three values, it is called
multi-class classification. This is the case for the Iris use case above.

Below, we illustrate an example of binary classification.

![img](https://inria.github.io/scikit-learn-mooc/figures/lin_separable.svg)

The data provided by the user contains 2 features, represented by the x- and
y-axis. This is a binary classification problem because the target contains
only 2 labels, here encoded by colors with blue and orange data points. Thus,
each data points represent a sample and the entire set was used to train a
linear model. The decision rule learned is thus the black dotted line. This
decision rule is used to predict the label of a new sample according the its
position with respect to the line: a sample lying on the left of the line will
be predicted as a blue sample while a sample lying on the right of the line
will be predicted as an orange sample. Here, we have a linear classifier
because the decision rule is defined as a line (it is called an hyperplane
in higher dimension). However, the shape of the decision rule will depend on
the model fitted.

### classifier

A model used for classification. These models handle targets that contains
discrete values such as `0`/`1` or `cat`/`dog`. For example in scikit-learn
`LogisticRegression` or `HistGradientBoostingClassifier` are classifier
classes.

Note: for historic reasons the `LogisticRegression` name is confusing.
`LogisticRegression` is not a regression model but a classification model.

### cross-validation

A procedure to estimate how well a model will generalize to new data. The main
idea behind this is to train a model on a dataset (called train set) and
evaluate its performance on a separate dataset (called test set).

This train/evaluate performance is repeated several times on different train
and test sets to get an estimate of the statistical model performance
uncertainties.

See
[this scikit-learn documentation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation)
for more details.

### data matrix, input data

In scikit-learn a common name for the data matrix is to call it `X` (following
the maths convention that matrices use capital letters and that input is call
`x` as in `y = f(x)`)

### estimator

In scikit-learn jargon: an object that has a `fit` method. The reasons for the
name estimator is that once the `fit` method is called on a model, the
parameters are learned (or estimated) from the data.

### feature, variable, attribute, descriptor, covariate

A quantity describing an observation (e.g. color, size, weight). You can see a
features as a quantity measured during the dataset collection.

For example, in the Iris dataset, features might include petal length and petal
width.

### hyperparameters

Aspects of model configuration that are not learnt from data. For example when
using a k-nearest neighbor approach, the number of neighbors to use is a
hyperparameter.

When trying to train a polynomial model (say of degree between 1 and 10 for
example) to 1 dimensional data, the degree of the polynomial is a
hyperparameter.

Hyperparameters will impact the statistical and computational performance
of a model. Indeed, hyperparameters of a model are usually inspected with
regard to their impact on the model performance and tuned to maximize model
performance (usually statistical performance). It is called hyperparameters
tuning and involve grid-search and randomized-search involving model evaluation
on some validation sets.

### infer/inference

This term might have different meaning in machine-learning and statistical
inference.

In machine-learning and more generally in this MOOC, we refer to inference the
process of making predictions by applying a trained model to unlabeled data. In
other words, inference is equivalent to predict the target of unseen data using
a fitted model.

In statistic inference, the notion of left-out/unseen data is not tight to
the definition. Indeed, inference refers to the process of fitting the
parameters of a distribution conditioned on some observed data. You can check
the Wikipedia article on
[statistical inference](https://en.wikipedia.org/wiki/Statistical_inference)
for more details.

### learned parameters

In scikit-learn the convention is that learned parameters finish with `\_` at
the end in scikit-learn (they are called attributes in scikit-learn glossary,
never used this and confusing with attributes = features). They are only
available after `fit` has been called.

Watch out parameters can also be used as a general Python meaning, as in
passing a parameter to a function or a class

### meta-estimator

In scikit-learn jargon: an estimator that takes another estimator as parameter.
Examples of meta-estimators include `Pipeline` and `GridSearchCV`.

### model

Generic term that refers to something that can learn prediction rules from the
data.

### overfitting

Overfitting occurs when your model stick too closely to the training data, so
that it ends up learning the noise in the dataset rather than the relevant
patterns. You can tell a model is overfitting when it performs great on your
train set, but poorly on your test set (or new real-world data).

### predictor

An estimator (object with a `fit` method) with a `predict` and/or `fit_predict`
method. Note a classifier or a regressor is a predictor. Example of predictor
classes are `KNeighborsClassifier` and `DecisionTreeRegressor`.

### predict/prediction

One of the focus of machine learning is to learn rules from data that we can
then use to make predictions on new samples that were not seen during training.

Example with a linear regression. If we do a linear regression in 1d and we
learn the linear model `y = 2 x - 5`. Say someone comes along and says what
does your model predict for `x = 10` we can use `y = 2*10 - 5 = 15`.

### regression

The goal is to predict a target that is continuous (contrary to discrete target
for classification problems). Example of regression problems are:

- predicting house prices from their descriptions (number of rooms, surface,
  location, etc ...)
- predicting the age of patients from their MRI scans

Below, we illustrate an example of regression.

![img](https://inria.github.io/scikit-learn-mooc/figures/dt_fit.svg)

The data provided by the user contains 1 feature called `x` and we want to
predict the continuous target `y`. Each black data points are samples used to
train a model. The model here is a decision tree and thus the decision rule
is defined as a piecewise constant function represented by the orange line.
To predict the target for a new sample for a given value of the x-axis, the
model will output the corresponding `y` value lying on the orange line.

### regressor

A regressor is a predictor in a regression setting.

In scikit-learn, `DecisionTreeRegressor` or `Ridge` are regressor classes.

### regularization / penalization

In linear models, regularization can be used in order to shrink the weights
towards zero. This can be useful to combat overfitting.

### sample, instance, observation

A data point in a dataset.

In the 2d data matrix, a sample is a row.

For example in the Iris dataset, a sample would be the measurements of a single
flower.

Note: "instance" is also used in a object-oriented meaning in this course. For
example, if we define `clf = KNeighborsClassifier()`, we say that `clf` is an
instance of the `KNeighborsClassifier` class.

### statistical performance / generalization performance / predictive performance

The performance of a model on the test data.

### supervised learning

TODO Talk about the general settings that we are trying to predict the target
given the features. reuse the phrasing from the intro.

`y = f(X)` y is the target, `X` is the data matrix, `f` is the model we are
trying to learn from the data.

A simple example in a 1d linear regression, we are trying to learn the model
`y = a*x + b`. The coefficients `a` and `b` are learned from the data, i.e.
adjusted so that the model fits the data as well as possible.

### target, label, annotation

The quantity we are trying to predict from the features. Labels are available
in a supervised learning setting and not in an unsupervised learning setting.

For example, in the Iris dataset, the features might include the petal length
and petal width, while the label would be the Iris species.

In scikit-learn convention: `y` is a variable commonly used to denote the
target. This is because the target can be seen as the output of the model and
follows the convention that output is called `y` as in `y = f(x)`.

### test set

The dataset used to evaluate the generalization performance of the model after
it is trained.

### train, learn, fit

Find ideal model parameters given the data.

TODO diagram from slides
![img](https://inria.github.io/scikit-learn-mooc/figures/linear_fit.svg)

For example if we have a 1d linear model like this: `y = a*x + b`. Training
this model on the data means finding the best line that is the closest to the
data points. In other words i.e. finding the best `a` (called slope) and `b`
(called intercept)

### train set

The dataset used to train the model.

### transformer

<https://scikit-learn.org/stable/glossary.html#term-transformer>

An [estimator](#estimator) (i.e. an object that has a `fit` method) supporting
`transform` and/or `fit_transform`. Examples for transformers are
`StandardScaler` or `ColumnTransformer`.

### underfitting

Underfitting occurs when your model does not have enough flexibility to
represent the data well. You can tell a model is underfitting when it performs
poorly on both training and test sets.

### unsupervised learning

In this setting, samples are not labelled. One particular example of
unsupervised learning is clustering, whose goal is to group the data into
subsets of similar samples. Potential applications of clustering include:

- using the content of articles to group them into broad topics
- finding different types of customers from a e-commerce website data

Note that although mentioned, unsupervised learning is not covered in this
course.

### validation set

TODO ??? (validation set is mentioned in some notebooks but probably never
defined)

different meaning validation set for early stopping, validation set for
optimizing hyperparameter train-validation-test set.

## Other useful glossaries

For generic machine learning terms:

- ML cheatsheet glossary: https://ml-cheatsheet.readthedocs.io/en/latest/glossary.html
- Google Machine Learning glossary:
  https://developers.google.com/machine-learning/glossary

For more advanced scikit-learn related terminology:

- scikit-learn glossary: https://scikit-learn.org/stable/glossary.html
