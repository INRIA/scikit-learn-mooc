# Quiz for each lesson

# lesson 0, ML concepts 
(from plan.md) Given a case study (e.g. pricing apartments based on a real estate website database) and sample toy dataset: 
- say whether it’s an application of supervised vs unsupervised, 
- classification vs regression, 
- what are the features, 
- what is the target variable, 
- what is a record.
- Propose a hand engineer decision rule that can be used as a baseline
- Propose a quantitative evaluation of the success of this decision rule.

# Lesson 1, exploratory analysis
cf inside the notebook

# lesson 2, basic preprocessing / first model with numerical feature
cf inside the notebook

# lesson 3, categorical feature
cf inside the notebook

# lesson 4, hyper-parameters tuning
cf inside the notebook

# Linear models
0. Could you explain what the LinearRegression's attribute `coef_` and `intercept_` is ?

1. At which conditions does linearRegression find the optimal error.
a) never 
b) if the data are linear
c) if the data are polynomial 
d) if the data are one dimensional
e) always

2. mean squared error is a measure:
a) for regression b) for classification
c) better when higher d) better when lower

3. Ridge model is:
a) same as LinearRegression with penalized weights
b) same as LogisticRegression with penalized weights
c) a Linear model
d) a non linear model

4. The parameter alpha in Ridge is learnt:
a) During the fit on the train set
b) by cross validation on the train set
c) by cross validation on the test set
d) could be choose by hand a priori

5. Which error/score could be used for classification:
a) Mean squared error
b) R2 score
c) Accuracy score

6. In a linear model, the number of parameters learnt are:
a) proportional to the number of samples
b) proportional to the number of features
c) fixed (chose by hand a priori)
d) proportional to the number of iteration

# Trees

1. How could we prevent overfitting within a tree model ?
a) with a weight regularization
b) by increasing max_depth
c) by decreasing max_depth
d) with early stopping

2. Tree are build:
a) incrementaly by splitting leafs
b) by refining the rules of each nodes
c) by refining the prediction of each leaf

2. to choose a split, we have to:
a) choose a random feature
b) choose a random value
c) maximize an equation
d) maximize nb_features equations

3. In regression setting, what represent a leaf in a decesion tree: 
a) a value
b) a value distribution
c) a class
d) probabilities of each class

3. In classification setting, what represent a leaf in a decesion tree: 
a) a value
b) a value distribution
c) a class
d) probabilities of each class

4. Decision tree could be used for:
a) regression
b) classification
c) clustering
d) dimension reduction

# Metrics

1. What is the accuracy score ?

2. What is true positive:
a) the number of positive prediction (sample classified positive by the model)
b) the number of positive label (sample with positive label)
c) the number of positive prediction which have positive label
d) the accuracy

2. What is false positive:
a) 1 - true positive
b) the number of positive prediction which have negative label
c) the number of negative prediction which have positive label
d) niether of the above

3. the confusion matrix is usefull if:
a) the class are imbalanced
b) the prediction are imbalanced
c) there are few samples
d) we want to get insight of the model failure

4. Precision correspond to:
a) true positive rate
b) accuracy
c) ratio of positive prediction over positive label
d) ratio of correct positive prediction over the number of positive prediction
e) ratio of correct prediction over the number of prediction

5. F1 is usefull if:
a) the accuracy is too high
b) class label are imbalanced
c) prediction are imbalanced
d) there are few samples

6. the area under the ROC curve:
a) characterize a dataset
c) characterize the performance of a classification model
c) characterize the performance of a regression model


# Ensemble

