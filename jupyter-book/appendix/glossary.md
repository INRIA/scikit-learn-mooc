# Glossary

This glossary aims to describe the main terms used in this course. For terms
that you don't find in this glossary, we added useful glossaries at the bottom
of this page.

## Main terms used in this course

### classification

Type of problems where the goal is to [predict](#predict-prediction) a
[target](#target-label-annotation) that can take finite set of values.

Examples of classification problems are:

- predicting the type of Iris (setosa, versicolor, virginica) from their petal
  and sepal measurements
- predicting whether patients has a particular disease from the result of their
  medical tests
- predicting whether an email is a spam or not from the email content, sender,
  title, etc ...

When the predicted [label](#target-label-annotation) can have two values, it is
called binary classification. This the case for the medical and spam use cases
above.

When the predicted [label](#target-label-annotation) can have at least three
values, it is called multi-class classification. This is the case for the Iris
use case above.

Below, we illustrate an example of binary classification.

![img](https://inria.github.io/scikit-learn-mooc/figures/lin_separable.svg)

The data provided by the user contains 2
[features](#feature-variable-attribute-descriptor-covariate), represented by
the x- and y-axis. This is a binary classification problem because the
[target](#target-label-annotation) contains only 2
[labels](#target-label-annotation), here encoded by colors with blue and orange
data points. Thus, each data points represent a
[sample](#sample-instance-observation) and the entire set was used to
[train](#train-learn-fit) a linear [model](#model). The decision rule learned
is thus the black dotted line. This decision rule is used to
[predict](#predict-prediction) the [label](#target-label-annotation) of a new
[sample](#sample-instance-observation) according its position with respect to
the line: a [sample](#sample-instance-observation) lying on the left of the
line will be predicted as a blue sample while a sample lying on the right of
the line will be predicted as an orange sample. Here, we have a linear
[classifier](#classifier) because the decision rule is defined as a line (in
higher dimensions this would be a hyperplane). However, the shape of the
decision rule will depend on the [model](#model) used.

### classifier

A model used for [classification](#classification). These models handle
[targets](#target-label-annotation) that contains discrete values such as
`0`/`1` or `cat`/`dog`. For example in scikit-learn `LogisticRegression` or
`HistGradientBoostingClassifier` are classification model classes.

Note: for historic reasons the `LogisticRegression` name is confusing.
`LogisticRegression` is not a regression model but a classification model, in
contrary with what the name would suggest.

### cross-validation

A procedure to estimate how well a [model](#model) will generalize to new data.
The main idea behind this is to [train](#train-learn-fit) a [model](#model) on
a dataset (called [train set](#train-set)) and evaluate its
[performance](#generalization-performance-predictive-performance-statistical-performance)
on a separate dataset (called [test set](#test-set)).

This train/evaluate performance is repeated several times on different train
and test sets to get an estimate of the model's generalization performance
uncertainties.

See [this scikit-learn
documentation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation)
for more details.

### data matrix, input data

The data containing only the
[features](#feature-variable-attribute-descriptor-covariate) and not the
[target](#target-label-annotation).

The data matrix has `n_samples` rows and `n_features` columns. For example for
the Iris dataset:

- the data matrix has a number of rows equal to the number of Iris flowers in
  the dataset
- the data matrix has 4 columns (for sepal length, sepal width, petal length,
  and petal width)

In scikit-learn a common name for the data matrix is to call it `X` (following
the maths convention that matrices use capital letters and that input is called
`x` as in `y = f(x)`)

### early stopping

This consists in stopping an iterative optimization method before the
convergence of the algorithm, to avoid over-fitting. This is generally done by
monitoring the generalization score on a [validation set](#validation-set).

### estimator

In scikit-learn jargon: an object that has a `fit` method. The reasons for the
name estimator is that once the `fit` method is called on a [model](#model),
the parameters are learned (or estimated) from the data.

### feature, variable, attribute, descriptor, covariate

A quantity describing a [sample](#sample-instance-observation) (e.g. color,
size, weight). You can see a
[features](#feature-variable-attribute-descriptor-covariate) as a quantity
measured during the dataset collection.

For example, in the Iris dataset, there are four
[features](#feature-variable-attribute-descriptor-covariate): sepal length,
sepal width, petal length and petal width.

### generalization performance, predictive performance, statistical performance

The performance of a [model](#model) on the [test data](#test-set). The [test
data](#test-set) where never seen by the [model](#model) during the
[training](#train-learn-fit) procedure.

### hyperparameters

Aspects of [model](#model) configuration that are not learnt from data.
Examples of hyperparameters:

- for a k-nearest neighbor approach, the number of neighbors to use is a
  hyperparameter
- for a polynomial model (say of degree between 1 and 10 for example), the
  degree of the polynomial is a hyperparameter.

Hyperparameters will impact the generalization and computational performance of a
model. Indeed, hyperparameters of a model are usually inspected with regard to
their impact on the model performance and tuned to maximize model performance
(usually
[generalization performance](#generalization-performance-predictive-performance-statistical-performance)
). It is called hyperparameters tuning and
involve grid-search and randomized-search involving model evaluation on some
[validation sets](#validation-set).

For more details, you can further read the following
[post](https://machinelearningmastery.com/difference-between-a-parameter-and-a-hyperparameter/)

### infer, inference

This term has a different meaning in machine-learning and statistical
inference.

In machine-learning and more generally in this MOOC, we refer to inference the
process of making [predictions](#predict-prediction) by applying a
[trained](#train-learn-fit) [model](#model) to unlabeled data. In other words,
inference is equivalent to [predict](#predict-prediction) the
[target](#target-label-annotation) of unseen data using a fitted
[model](#model).

In statistic inference, the notion of [left-out/unseen data](#test-set) is not
tied to the definition. Indeed, inference refers to the process of
[fitting](#train-learn-fit) the parameters of a distribution conditioned on
some observed data. You can check the Wikipedia article on
[statistical inference](https://en.wikipedia.org/wiki/Statistical_inference)
for more details.

### learned parameters

In scikit-learn the convention is that learned parameters finish with `\_` at
the end in scikit-learn. They are only available after
[`fit`](#train-learn-fit) has been called.

An example for such a parameter are the slope and intercept of a linear
[model](#model) in one dimension see this [section](#train-learn-fit) for more
details about such a model.

Note: parameters can also be used in a general Python meaning, as in passing a
parameter to a function or a class

### meta-estimator

In scikit-learn jargon: an [estimator](#estimator) that takes another
[estimator](#estimator) as parameter. Examples of meta-estimators include
`Pipeline` and `GridSearchCV`.

### model

Generic term that refers to something that can [learn](#train-learn-fit)
[prediction](#predict-prediction) rules from the data.

### overfitting

Overfitting occurs when your [model](#model) stick too closely to the [training
data](#train-set), so that it ends up learning the noise in the dataset rather
than the relevant patterns. You can tell a [model](#model) is overfitting when
it performs great on your [train set](#train-set), but poorly on your [test
set](#test-set) (or new real-world data).

### predictor

An [estimator](#estimator) (object with a `fit` method) with a `predict` and/or
`fit_predict` method. Note a [classifier](#classifier) or a
[regressor](#regressor) is a predictor. Example of predictor classes are
`KNeighborsClassifier` and `DecisionTreeRegressor`.

### predict, prediction

One of the focus of machine learning is to learn rules from data that we can
then use to make predictions on new [samples](#sample-instance-observation)
that were not seen during [training](#train-learn-fit).

Example with a linear [regression](#regression). If we do a linear
[regression](#regression) in 1d and we learn the linear [model](#model)
`y = 2*x - 5`. Say someone comes along and says what does your [model](#model)
predict for `x = 10` we can use `y = 2*10 - 5 = 15`.

### regression

The goal is to [predict](#predict-prediction) a
[target](#target-label-annotation) that is continuous (contrary to discrete
[target](#target-label-annotation) for [classification](#classification)
problems). Example of regression problems are:

- predicting house prices from their descriptions (number of rooms, surface,
  location, etc ...)
- predicting the age of patients from their MRI scans

Below, we illustrate an example of regression.

![img](https://inria.github.io/scikit-learn-mooc/figures/dt_fit.svg)

The data provided by the user contains 1
[feature](#feature-variable-attribute-descriptor-covariate) called `x` and we
want to [predict](#predict-prediction) the continuous
[target](#target-label-annotation) `y`. Each black data points are
[samples](#sample-instance-observation) used to [train](#train-learn-fit) a
[model](#model). The [model](#model) here is a decision tree and thus the
decision rule is defined as a piecewise constant function represented by the
orange line. To [predict](#predict-prediction) the
[target](#target-label-annotation) for a new
[sample](#sample-instance-observation) for a given value of the x-axis, the
[model](#model) will output the corresponding `y` value lying on the orange
line.

### regressor

A regressor is a [predictor](#predictor) in a [regression](#regression)
setting.

In scikit-learn, `DecisionTreeRegressor` or `Ridge` are regressor classes.

### regularization, penalization

In linear [models](#model), regularization can be used in order to
shrink/constrain the weights/parameters towards zero. This can be useful to
avoid [overfitting](#overfitting).

### sample, instance, observation

A data point in a dataset.

In the 2d [data matrix](#data-matrix-input-data), a sample is a row.

For example in the Iris dataset, a sample would be the measurements of a single
flower.

Note: "instance" is also used in a object-oriented meaning in this course. For
example, if we define `clf = KNeighborsClassifier()`, we say that `clf` is an
instance of the `KNeighborsClassifier` class.

### supervised learning

We can give a concrete graphical example.

![img](https://inria.github.io/scikit-learn-mooc/figures/boosting0.svg)

The plot represent a [supervised](#supervised-learning)
[classification](#classification) example. The data are composed of 2 features
since we can plot each [data point](#sample-instance-observation) on a 2-axis
plot. The color and shape correspond to the [target](#target-label-annotation)
and we have 2 potential choices: blue circle vs. orange square.

Supervised learning learning boiled down to the fact that we have access to the
target. During fitting, we exactly know if a [data
point](#sample-instance-observation) will be a blue circle or an orange square.

In the contrary [unsupervised learning](#unsupervised-learning) will only have
access to the [data points](#sample-instance-observation) and not the target.

Framing a machine learning problem as a [supervised](#supervised-learning) or
[unsupervised](#unsupervised-learning) learning problem will depend of the data
available and the data science problem to be solved.

### target, label, annotation

The quantity we are trying to [predict](#predict-prediction) from the
[features](#feature-variable-attribute-descriptor-covariate). Targets are
available in a [supervised learning](#supervised-learning) setting and not in
an [unsupervised learning](#unsupervised-learning) setting.

For example, in the Iris dataset, the
[features](#feature-variable-attribute-descriptor-covariate) might include the
petal length and petal width, while the label would be the Iris specie.

In scikit-learn convention: `y` is a variable name commonly used to denote the
target. This is because the target can be seen as the output of the
[model](#model) and follows the convention that output is called `y` as in `y = f(x)`.

Target is usually used for [regression](#regression) setting while label is
usually used in [classification](#classification) setting.

### test set

The dataset used to make predictions of a [model](#model) after it is
[trained](#train-learn-fit) and eventually evaluate its [generalization
performance](#generalization-performance-predictive-performance-statistical-performance).

### train, learn, fit

Find ideal [model](#model) parameters given the data. Let's give a concrete
example.

![img](https://inria.github.io/scikit-learn-mooc/figures/linear_fit_red.svg)

On the above figure, a linear [model](#model) (blue line) will be
mathematically defined by `y = a*x + b`. The parameter `a` defines the slope of
the line while `b` defines the intercept. Indeed, we can create an infinity of
models by varying the parameters `a` and `b`. However, we can search for a
specific linear [model](#model) that would fulfill a specific requirement, for
instance minimizing the sum of the errors (red lines). Training, learning, or
fitting a [model](#model) refers to the procedure that will find the best
possible parameters `a` and `b` fulfilling this requirement.

In a more abstract manner, we can represent fitting with the following diagram:

![img](https://inria.github.io/scikit-learn-mooc/_images/api_diagram-predictor.fit.svg)

The model state are indeed the parameters and the jockey wheels are referring to
an optimization algorithm to find the best parameters.

### train set

The dataset used to train the [model](#model).

### transformer

An [estimator](#estimator) (i.e. an object that has a `fit` method) supporting
`transform` and/or `fit_transform`. Examples for transformers are
`StandardScaler` or `ColumnTransformer`.

### underfitting

Underfitting occurs when your [model](#model) does not have enough flexibility
to represent the data well. You can tell a [model](#model) is underfitting when
it performs poorly on both [training](#train-set) and [test](#test-set) sets.

The opposit of underfitting is [overfitting](#overfitting).

### unsupervised learning

In this setting, samples are not labelled. One particular example of
unsupervised learning is clustering, whose goal is to group the data into
subsets of similar samples. Potential applications of clustering include:

- using the content of articles to group them into broad topics
- finding different types of customers from a e-commerce website data

Note that although mentioned, unsupervised learning is not covered in this
course. The opposite of unsupervised learning is [supervised
learning](#supervised-learning).

### validation set

A machine learning model is evaluated on the following manner: the
[model](#model) is [trained](#train-learn-fit) using a [training
set](#train-set) and evaluated using a [testing set](#test-set). In this
setting, it is implied that the [hyperparameters](#hyperparameters) of the
[model](#model) are fixed.

When one would like to tune the [hyperparameters](#hyperparameters) of a
[model](#model) as well, then it is necessary to subdivide the [training
set](#train-set) into a training and a validation set: we fit several machine
learning [models](#model) with different [hyperparameters](#hyperparameters)
values and select the one performing best on the validation set. Finally, once
the [hyperparameters](#hyperparameters) fixed we can use the left-out [testing
set](#test-set) to evaluate this model.

Sometimes, we also use a validation set in context of
[early-stopping](#early-stopping). It is used with machine learning using
iterative optimization to be fitted and it is not clear how many iterations are
needed to train the model. In this case, one will used a validation set to
monitor the performance of the model on some data different from the [training
set](#train-set). Once that some criteria are fulfilled, the model is trained.
This model is finally evaluated on the left-out [testing set](#test-set).

## Other useful glossaries

For generic machine learning terms:

- ML cheatsheet glossary:
  https://ml-cheatsheet.readthedocs.io/en/latest/glossary.html
- Google Machine Learning glossary:
  https://developers.google.com/machine-learning/glossary

For more advanced scikit-learn related terminology:

- scikit-learn glossary: https://scikit-learn.org/stable/glossary.html
