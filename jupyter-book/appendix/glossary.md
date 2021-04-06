# Glossary

This glossary is supposed to describe the main terms used in this course. For
terms that you can not find here, we added useful glossaries at the bottom of
this page.

## Main terms used in this course

### classification

type of problems where the goal is to predict a target that can take finite set
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

TODO basically the idea is to use all the terms on a given example. This is
simpler than explaining each term individually. for features, samples,
prediction, decision rule

### classifier

a model used for classification. For example in scikit-learn
`LogisticRegression` or `HistGradientBosstingClassifier` are classifier
classes.

Note: for historic reasons the `LogisticRegression` name is confusing.
`LogisticRegression` is not a regression model but a classification model.

### cross-validation

A procedure to estimate how well a model will generalize to new data. The main
idea behind this is to train a model on a dataset (called train set) and
evaluate its performance on a separate dataset (called test set). See
[this](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation)
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

A quality describing an observation (e.g. color, size, weight)

For example, in the Iris dataset, features might include petal length and petal
width.

### hyperparameters

Aspects of model configuration that are not learnt from data. For example when
using a k-nearest neighbor approach, the number of neighbors to use is a
hyper-parameter.

When trying to train a polynomial model (say of degree between 1 and 10 for
example) to 1 dimensional data, the degree of the polynomial is a
hyper-parameter.

TODO Say something about validation that basically validation is used to find
the best hyperparameters ???


### infer/inference

TODO we only mention it in the intro slides in a statistical meaning, we
probably use it in the notebook as a verb.

mention statistical meaning vs machine-learning usage (from google glossary
below:)

In machine learning, often refers to the process of making predictions by
applying the trained model to unlabeled examples. In statistics, inference
refers to the process of fitting the parameters of a distribution conditioned
on some observed data. (See the Wikipedia article on statistical inference.)


### learned parameters

In scikit-learn the convetion is that learned parameters finish with `\_` at
the end in scikit-learn (they are called attributes in scikit-learn glossary,
never used this and confusing with attributes = features). They are only
available after `fit` has been called.

watch out parameters can also be used as a general Python meaning, as in
passing a parameter to a function or a class


### meta-estimator

In scikit-learn jargon: an estimator that takes another estimator as parameter.
Examples of meta-estimators include `Pipeline` and `GridSearchCV`.


### model

generic term that refers to something that can learn prediction rules from the
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

problem the goal is to predict a target that is continuous. Example of
regression problems are:

- predicting house prices from their descriptions (number of rooms, surface,
  location, etc ...)
- predicting the age of patients from their MRI scans

TODO

Reuse https://inria.github.io/scikit-learn-mooc/figures/linear_fit.svg or
linear regression 1d e.g. with Penguin example???

basically the idea is to use all the terms on a given example. This is simpler
than explaining each term individually. for features, samples, prediction,
decision rule

### regressor
A regressor is a predictor in a regression setting.

In scikit-learn, `DecisionTreeRegressor` or `Ridge` are regressor classes.

### regularization / penalization

In linear models, regularization can be used in order to shrink the weights
towards zero. This can be useful to combat overfitting.


### sample, instance, observation

a data point in a dataset.

In the 2d data matrix, a sample is a row.

For example in the Iris dataset, a sample would be the measurements of a single
flower.

Note: "instance" is also used in a object-oriented meaning in this course. For
example, if we define `clf = KNeighborsClassifier()`, we say that `clf` is an
instance of the `KNeighborsClassifier` class.

### supervised learning

TODO Talk about the general settings that we are trying to predict the target
given the features. reuse the phrasing from the intro.

`y = f(X)` y is the target, `X` is the data matrix, `f` is the model we are
trying to learn from the data.

A simple example in a 1d linear regression, we are trying to learn the model `y
= a*x + b`. The coefficients `a` and `b` are learned from the data, i.e.
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

An estimator (i.e. an object that has a `fit` method) supporting `transform`
and/or `fit_transform`. Examples for transformers are `StandardScaler` or
`ColumnTransformer**

### underfitting

Underfitting occurs when your model does not have enough flexibility to
represent the data well. You can tell a model is underfitting when it performs
poorly on both training and test sets.

### unsupervised learning

In this setting, samples are not labelled. One particular example of
unsupervised learning is clustering, whose goal is to group the data into
subsets of similar samples. Potential applications of clustering include:

-   using the content of articles to group them into broad topics
-   finding different types of customers from a e-commerce website data

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
