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
partners, and movies. It has also become indispensable to many empirical
sciences, including physics, astronomy, biology, and the social sciences.

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
- What is machine learning? (Sample applications)
- Kinds of machine learning: unsupervised vs supervised.
- Data formats and preparation.

- Supervised learning: Interface
- Supervised learning: Training and test data
- Supervised learning: Classification
- Supervised learning: Regression
- Unsupervised Learning: Unsupervised transformers
- Unsupervised Learning: Preprocessing and scaling
- Unsupervised Learning: Dimensionality reduction
- Unsupervised Learning: Clustering
- Summary : Estimator interface

- Application: Classification of digits
- Methods: Unsupervised learning
- Application : Eigenfaces
- Methods: Text feature abstraction, bag of words
- Application : Insult detection
- Summary : Model building and generalization

Afternoon Session
------------------
- Cross-Validation
- Model Complexity: Overfitting and underfitting
- Complexity of various model types
- Grid search for adjusting hyperparameters 

- Basic regression with cross-validation
- Application : Titanic survival with Random Forest

- Building Pipelines: Motivation and Basics
- Building Pipelines: Preprocessing and Classification
- Building Pipelines: Grid-searching Parameters of the feature extraction
- Application : Image classification

- Model complexity, learning curves and validation curves
- In-Depth: Linear Models
- In-Depth: Kernel SVMs
- In-Depth: trees and Forests

- Learning with Big Data: Out-Of-Core learning
- Learning with Big Data: The hashing trick for large text corpuses
