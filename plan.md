Note: to edit on HackMD https://hackmd.io/6fFy3_y6SOWonOYeGaNiHw?edit

# Intro module: What is Machine Learning
 (only slides / videos i.e. no code)
## Learning objectives:

- familiarity with scope of most common ML applications in business or science
- understand difference between memorization and generalization
- familiarity with main machine learning concepts and vocabulary

## Content
Why and when? Example applications.
- Mention the iris example, pitch it as historical, but also as a
  botanical and agriculture problem.
  The benefit of this example is that it forces to think about
  measurement, but also because it has one class that is easy to separate
  only one feature
    - Aurelie: real irises for the video?
- The "adult" dataset
    - Maybe looking at it with excel, to be in an environment familiar to
      people
    - Mention the importance of data visualization: intuitions about the
      data can be very helpful

Learning from data vs expertly engineered decision rules
- One the iris example, show that cutting on one specific feature
  separates well one class
- How do we automate this? How do we achieve this on more complex data
  such as the census dataset?

Descriptive vs predictive analysis
- Generalization (Out of sample properties)
- An example of where it makes a difference: if the data has redundant
  variables, such as expressing the education level as the name of the
  degree or the corresponding number of years of education

Generalization vs memorization: the need for a train / test split
- The nearest neighbors example to illustrate this

Supervised vs Unsupervised
- Formalize supervised learning (define "X" and "y")
- Introduce unsupervised learning, for instance dimensionality reduction
  (and go back to the example of redundant variables: if we have many of
  these, we should be able to reduce the problem without even looking at
  y

Regression vs Classification
- In the adult data: it would make more sense to do a continuous
  prediction
- In the iris example, it is naturally a classification problem

Features and samples
- The data matrix
  - Build the data matrix of Iris

A few words about the style and scope of this course: it is centered
around code, though we strive to keep it simple

## Quizz:
Given a case study (e.g. pricing apartments based on a real estate website database) and sample toy dataset: say whether it’s an application of supervised vs unsupervised, classification vs regression, what are the features, what is the target variable, what is a record.

Propose a hand engineer decision rule that can be used as a baseline

Propose a quantitative evaluation of the success of this decision rule.



# The Predictive Modeling Pipeline

## Notebook module #1: exploratory analysis
### Learning objectives:
    
- load tabular data with pandas
- visualize marginal distribution with histograms
- visualize pairwise interactions with scatter plots
- identify outlier and dynamic range of each column

### Content
Defining a predictive task that relates to the business or scientific case

Pandas read_csv

Simple exploratory data analysis with pandas and matplotlib


## Notebook module #2: basic preprocessing for minimal model fit

### Learning objectives:
- Know the difference between a numerical and a categorical variable
- use a scaler
- convert category labels to dummy variables
- combine feature preprocessing and model with pipeline
- evaluate generalization of model with cross-validation

### Content

Prepare a train  / test split

Basic model on numerical features only

Basic processing: missing values and scaling

Use a pipeline to evaluate model with cross-validation with and without scaling

Handling categorical variables with one-hot encoding

Use the column transformer to build pipeline with heterogeneous dtype

Model fitting and performance evaluation with cross-validation

- Gael thinks that we could use a video here for cross-validation
  (in particular, the "plot_cv_indices" in the notebook gets a bit in the
  way of being accessible and didactic

## Notebook module #3: basic parameter tuning and final test score evaluation

### Learning objectives:
- Learn to no trust blindly the default parameters of scikit-learn estimators

### Content
Parameter tuning with Grid and Random hyperparameter search
Nested cross-validation

Confirmation of performance with final test set

# Supervised learning
## Learning objectives:
Understand decision rules for a few important algorithms
Know how to diagnose model generalization errors (overfitting especially)
How to use variable selection and generalization to fight overfitting
Feature engineering to limit underfitting

## Olivier: Overfitting/Underfitting validation curves, learning curves, regularisation with linear models

- Video about overfitting?

## Loïc: Trees in depth + ensembles

## Guillaume: Evaluation of supervised learning models:
        
Confusion matrix for classifiers / precision / recall / ROC AUC curve (Mention imbalanced classes)
Predict vs True plot for regressors

## Olivier: Linear models in depth
Logistic Regression, linear regression, classification vs regression, multi-class, linear separability. Pros and cons
        L1 and L2 penalty for linear models
        Learning curves and validation curves (video: how to read curves)

## Baselines: majority class classifier (already in second module) and k-nearest neighbors

## Feature engineering to augment the expressivity of linear models:

Binning / Polynomial feature extraction / Nystroem method
        
Feature selection to combat overfitting and speed-up models

## Univariate feature selection
Show catastrophic example where feature selection is done on the whole dataset rather than only on train


## Evaluating the feature importance with permutations
Failure Mode : cardinality bias of overfitting random forest feature importances

## Looking at the decision function with partial dependence plots

Gael thinks that explaining the difference between conditional and
marginal interpretation is important.

Stability of hyperparameter during cross-validation

