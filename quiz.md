# Quiz for each lesson

## lesson 0, ML concepts

Given a case study: pricing apartments based on a real estate website. We have the record of thousand house descriptions with their price. But for some house the price is not mentioned, and we want to predict it.

1. What kind of problem is it?
a) a supervised problem
b) an unsupervised problem
c) a classification problem
d) a regression problem

_solution_ a) & d) It is a supervised problem because we have some information about the target variable (the price). It is a regression problem because the target variable is continous (it is not a class)

2. What are the features?
a) the number of rooms might be a feature
b) the localisation of the house might be a feature
c) the price of the house might be a feature

_solution_ a) & b) Every kind of house description might be a feature here.

3. What is the target variable?
a) The text descritpion is the target
b) the price of the house is the target
c) the house with no price mentioned are the target

_solution_ b) The price is the amount we want to predict, thus it is our target variable

4. What is a record (a sample)?
a) each house description is a record
b) each house price is a record
c) each kind of description (as the house size) is a record

_solution_ a)

- Propose a hand engineer decision rule that can be used as a baseline
- Propose a quantitative evaluation of the success of this decision rule.

## Lesson 1, exploratory analysis

1. `import pandas as pd` allows us to:
a) deals with CSV files
b) deals with tabular data
c) plot basic information about tabular data
d) deals with scientific/mathematics functions

_solution_ a) & b) & c)

2. a *categorical* variable:
a) is a variable with only two different values
b) is a variable with continuous numerical values
c) is a variable with a finit set of value

_solution_ c)

## lesson 2, basic preprocessing / first model with numerical feature

cf inside the notebook

## lesson 3, categorical feature

cf inside the notebook

## lesson 4, hyper-parameters tuning

cf inside the notebook

## Linear models

0. Could you explain what the LinearRegression's attribute `coef_` and `intercept_` is?

_solution_ The prediction of a linear regression model is as follow: y = `coef_` * X + `intercept_`

1. At which conditions does linearRegression find the optimal error.
a) never
b) if the data are linear
c) if the data are polynomial
d) if the data are one dimensional
e) always

_solution_ e) linearRegression optimize only the error, while Ridge has a trade off between the error and the regularization of the model

2. mean squared error is a measure:
a) for regression b) for classification
c) better when higher d) better when lower

_solution_ a) & d) contrary to a score, an error is always better when lower

3. Ridge model is:
a) same as LinearRegression with penalized weights
b) same as LogisticRegression with penalized weights
c) a Linear model
d) a non linear model

_solution_ a) & c) LogisticRegression already come with a regularization parameter

4. The parameter alpha in Ridge is:
a) Learnt during the fit on the train set
b) choose by cross validation on the train set
c) choose by cross validation on the test set
d) could be choose by hand a priori

_solution_ b) & d) Only `coef_` and `intercept_` are fitted during the fit. One could never choose any hyper-parameters on the test set (it is overfitting)

5. Which error/score could be used for classification:
a) Mean squared error
b) R2 score
c) Accuracy score

_solution_ c) Mean squared error and R2 score are for regression

6. In a linear model, the number of parameters learnt are:
a) proportional to the number of samples
b) proportional to the number of features
c) fixed (chose by hand a priori)
d) proportional to the number of iteration

_solution_ b) There a exactly number of features + 1 parameters in a linear model

## Trees

1. How could we prevent overfitting within a tree model ?
a) with a weight regularization
b) by increasing max_depth
c) by decreasing max_depth
d) with early stopping

_solution_ c) There is no such weight regularization or early stopping in decision tree.

2. Tree are build:
a) incrementally by splitting leafs
b) by refining the rules of each nodes
c) by refining the prediction of each leaf

_solution_ a)

2. In a decision tree, to choose a split, we have to:
a) randomly choose a feature
b) randomly choose a value
c) maximize an equation
d) maximize nb_features equations

_solution_ d) At each step, it compute the max information gain for each feature

3. In regression setting, what represent a leaf in a decision tree:
a) a value
b) a value distribution
c) a class
d) probabilities of each class

_solution_ a) Each leaf correspond to a single value

3. In classification setting, what represent a leaf in a decision tree:
a) a value
b) a value distribution
c) a class
d) probabilities of each class

_solution_ c) Each leaf predicts a single class. However, in random forest, we use the probabilities of each class to aggregate prediction

4. Decision tree could be used for:
a) regression
b) classification
c) clustering
d) dimension reduction

_solution_ a) & b)

## Metrics

1. What is the accuracy score ?

_solution_ The ratio of correct predictions over the number of predictions

2. What is true positive:
a) the number of positive prediction (sample classified positive by the model)
b) the number of positive label (sample with positive label)
c) the number of positive prediction which have positive label
d) the accuracy

_solution_ c)

2. What is false positive:
a) 1 - true positive
b) the number of positive prediction which have negative label
c) the number of negative prediction which have positive label
d) neither of the above

_solution_ b)

3. the confusion matrix is useful if:
a) the class label are imbalanced
b) there are few features
c) there are few samples
d) we want to get insight of the model failure

_solution_ a) & d)

4. Precision correspond to:
a) true positive rate
b) accuracy
c) ratio of positive prediction over positive label
d) ratio of correct positive prediction over the number of positive prediction
e) ratio of correct prediction over the number of prediction

_solution_ d)

5. F1 is useful if:
a) the accuracy is too high
b) class label are imbalanced
c) there are few samples

_solution_ b)

6. the area under the ROC curve:
a) characterize a dataset
b) characterize the performance of a classification model
c) characterize the performance of a regression model

_solution_ b)

## Ensemble

## Feature selection

## Feature importance

1. With a same dataset, feature importance might differs if:
a) we use two different models
b) we use two different train/test split with a same model
c) we use a same model with a different set of hyper-parameters
d) we use a same model with the same set of hyper-parameters but a different random_state

_solution_ a) & b) & c) & d) The feature importance depends of the model, the split and the hyper-parameters

2. In linear model, the feature importance:
a) might be infer from the coefficients
b) might be infer by `importance_permutation`
c) need a regularization to infer the importance
d) is a built-in attribute

_solution_ a) & b)

3. If two feature are the same (thus correlated)
a) their feature importance will be the same
b) their feature importance will be divided by 2
c) only one will receive all the feature importance, the second one will be 0
d) it depends

_solution_ d) it depends how the model will use them.

5. RandomForest.feature_importances_
a) has bias for categorical feature
b) has bias for continous (high cardinality) feature
c) is independant from the train/test split
d) is independant from the hyper-parameters

_solution_ b)

6. To evaluate the feature importance for a specific model, one could:
a) drop a column and compare the score
b) shuffle a column and compare the score
c) put all column to 0 and compare the score
d) change a column value to random number and compare the score

_solution_ b) Droping a column will change the model. Changing the column value to 0 or random number might completly change its distribution (thus the model might behave unexpectedly)
