Machine Learning with scikit-learn

Tutorial Topic
--------------

This tutorial aims to provide an introduction to machine learning and
scikit-learn "from the ground up". We will start with core concepts of machine
learning, some example uses of machine learning, and how to implement them
using scikit-learn. Going in detail through the characteristics of several
methods, we will discuss how to pick an algorithm for your application, how to
set its parameters, and how to evaluate performance.

Please provide a more detailed abstract of your tutorial (again, see last years tutorials).
---------------------------------------------------------------------------------------------

Machine learning is the task of extracting knowledge from data, often with the
goal of generalizing to new and unseen data. Applications of machine learning 
now touch nearly every aspect of everyday life, from the face detection in our
phones and the streams of social media we consume to picking restaurants,
partners, and movies. Machine learning has also become indispensable to many
empirical sciences, from physics, astronomy and biology to social sciences.

Scikit-learn has emerged as one of the most popular toolkits for machine
learning, and is now widely used in industry and academia.
The goal of this tutorial is to enable participants to use the wide variety of
machine learning algorithms available in scikit-learn on their own data sets,
for their own domains.

This tutorial will comprise an introductory morning session and an advanced
afternoon session. The morning part of the tutorial will cover basic concepts
of machine learning, data representation, and preprocessing. We will explain
different problem settings and which algorithms to use in each situation.
We will then go through some sample applications using algorithms implemented
in scikit-learn, including SVMs, Random Forests, K-Means, PCA, t-SNE, and
others.

In the afternoon session, we will discuss setting hyper-parameters and how to
prevent overfitting. We will go in-depth into the trade-off of model complexity
and dataset size, as well as discussing complexity of learning algorithms and
how to cope with very large datasets. The session will conclude by stepping
through the process of building machine learning pipelines consisting of
feature extraction, preprocessing and supervised learning.


Outline
========

Morning Session
----------------

- Introduction to machine learning with sample applications

- Types of machine learning: Unsupervised vs. supervised learning

- Scientific Computing Tools for Python: NumPy, SciPy, and matplotlib

- Data formats, preparation, and representation

- Supervised learning: Training and test data
- Supervised learning: The scikit-learn estimator interface
- Supervised learning: Estimators for classification
- Supervised learning: Estimators for regression analysis

- Unsupervised learning: Unsupervised Transformers
- Unsupervised learning: Preprocessing and scaling
- Unsupervised learning: Feature extraction and dimensionality reduction
- Unsupervised learning: Clustering

- Preparing a real-world dataset
- Working with text data via the bag-of-words model
- Application: IMDB Movie Review Sentiment Analysis


Afternoon Session
------------------
- Cross-Validation
- Model Complexity: Overfitting and underfitting
- Complexity of various model types
- Grid search for adjusting hyperparameters 

- Scikit-learn Pipelines

- Supervised learning: Performance metrics for classification
- Supervised learning: Support Vector Machines
- Supervised learning: Algorithm and model selection via nested cross-validation
- Supervised learning: Decision trees and random forests, and ensemble methods

- Unsupervised learning: Non-linear regression analysis
- Unsupervised learning: Hierarchical and density-based clustering algorithms
- Unsupervised learning: Non-linear dimensionality reduction

- Wrapper, filter, and embedded approaches for feature selection

- Supervised learning: Artificial neural networks: Multi-layer perceptrons
- Supervised learning: Out-of-core learning
