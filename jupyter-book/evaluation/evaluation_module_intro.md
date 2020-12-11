# Module overview

## What you will learn

<!-- Give in plain English what the module is about -->

In the previous module, we presented the general cross-validation framework
and used it to evaluate models' performance. However, this is important to
keep in mind that some elements in the cross-validation need to be decided
depending of the nature of the problem: (i) the cross-validation strategy and
(ii) the evaluation metrics. Besides, it is always good to compare the models'
performance with some baseline for which one expects a given level of
performance.

In this module, we present both aspects and give insights on when to use a
specific cross-validation strategy and a metric. In addition, we will also
give some insights regarding how to compare a model with some baseline.

## Before getting started

<!-- Give the required skills for the module -->

The required technical skills to carry on this module are:

- know Python;
- having some basic knowledge of the following libraries: NumPy, SciPy,
  Pandas, Matplotlib, and Seaborn;
- basic understanding of predictive scikit-learn pipeline.

<!-- Point to resources to learning these skills -->

The following links will help you to have an introduction to the above
required libraries:

- [Introduction to Python](https://scipy-lectures.org/intro/language/python_language.html);
- [Introduction to NumPy](https://scipy-lectures.org/intro/numpy/index.html);
- [Introduction to Matplotlib](https://scipy-lectures.org/intro/matplotlib/index.html);
- [Introduction to SciPy](https://scipy-lectures.org/intro/scipy.html);
- [Introduction to Pandas](https://pandas.pydata.org/docs/user_guide/10min.html#min);
- [Introduction to Seaborn](https://seaborn.pydata.org/introduction.html).

## Objectives and time schedule

<!-- Give the learning objectives -->

The objective in the module are the following:

- understand the necessity of using an appropriate cross-validation strategy
  depending on the data;
- get the intuitions behind comparing a model with some basic models that
  can be used as baseline;
- understand the principle behind using nested cross-validation when the model
  needs to be evaluated as well as optimized;
- understand the difference between regression and classification metrics;
- understand the difference between metrics.

<!-- Give the investment in time -->

The estimated time to go through this module is about 4 hours.
