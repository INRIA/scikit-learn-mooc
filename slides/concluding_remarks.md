class: center, middle

# MOOC Machine learning with scikit-learn

<img src="../figures/scikit-learn-logo.svg" style="width: 90%;">

???

Hi, welcome to the last video of the MOOC on machine learning with scikit-learn

---

class: titlepage

.header[MOOC Machine learning with scikit-learn]

# Concluding remarks

---

class: middle

- A lot of material so far

- Congratulations!

- And thank you to everyone involved

???

We have covered a lot of material in this MOOC so far.

Congratulations for getting here! We know that this quite an investment.

And thank you to everyone involved, the instructors, all the support
staff, the people who helped on the forum, and you, the students, for the
hard work!

---

# Stepping back

- The big messages of the MOOC
- Going further with machine learning
- Beyond machine-learning: The bigger picture


---
class: middle

# The big messages of the mooc

???

Let me start by summarizing the what we have learned about machine
learning with scikit-learn.

---

# Machine learning in a nutshell

- Learned on a train set; applied to test set

???

Machine learning in a nutshell, we have seen that

- A predictive model is learned on a train set and then applied to new data, a "test set"

--

- Built from a data matrix, a given number of features for each observation

???

- Scikit-learn models are built from a data matrix, of a given number of features for each observation

--

- Transformations of the data
    - Encoding of the categorical variables
    - Only using information available at train time
    - The scikit-learn Pipeline object

???

- Transformations of the data, such as encoding of the categorical variables, prior the predictive model, are often important.
    - Care must be done to use only information available at train time.
    - For this, you need to use a scikit-learn Pipeline object to chain the data transformation with the predictive model


---

# Machine learning in a nutshell

- Minimize the error on the test set
    - train error can detect underfit: models too simple for the data

???

- The best predictive model is selected to minimize the error on the test set, but monitoring the train error can be useful to detect underfit: models too simple for the data

--

- Multiple hyper-parameters
    - Control model complexity
    - Selecting hyper-parameters is important
    - GridSearchCV, RandomSearchCV...

???

- Models in scikit-learn come with multiple hyper-parameters
    - These can control model complexity.
    - Selecting model hyper-parameter is important.
    - It can be done with objects such as GridSearchCV, RandomSearchCV and the like


---

# Machine learning in a nutshell

- Understanding the models
    - know when they are suited to the data 
    - intuitions on how to debug them

???

- Understanding the models, in the sens of how the build their predictive functions, helps knowing when they are suited to the data and having intuitions on how to debug them

--

- Linear models
  - Combining the values of features
  - For many features or few observations

???

- Linear models make decisions by combining the values of each feature. They are particularly suited when the number of features is large or the number of observations is small

--

- Tree-based
    - A series of binary choices (thresholds)
    - For tabular data, columns are quantity of different nature 
    - Gradient boosted trees (HistGradientBoostingRegressor and Classifier) are goto methods

???

- Tree-based models
  - combine a series of binary choices, such as thresholds on the values of 
    various attributes.
  - They are particularly suited to tabular data where columns are quantity of different nature (age, height, job title), a mixing of categorical and continuous
variables, or have missing values.
  - The gradient boosted trees (HistGradientBoostingRegressor and Classifier) are typically goto methods when there are more than a few thousands of samples

---
class: middle

# Going further with machine learning

???

Now, let me give a few pointers on going further with machine learning.

---


# Learning more about scikit-learn

- The scikit-learn doc
    - The documentation is rich, didactic, continuously improving
- These docs comprise
  - An user guide
  - API docs
  - Examples
- Where to ask questions:
    - Stackoverflow

---

# We are an open-source community

- Free, open, driven by a community spirit, trying to be inclusive
- How to contribute
    - Build a community: helping each other, helping training, communication, advocacy
    - We are developed on github:
	- Curate information: one of our challenges is the information overflow
	- Contributing code is technical
	    - Learn software engineering:
		- Learn git, learn the github workflow
		- https://lab.github.com/

---

# Studying machine learning further

- [Introduction to Machine Learning with Python](https://www.oreilly.com/library/view/introduction-to-machine/9781449369880/) by Andreas C. Müller, Sarah Guido
  
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) by Jake van der Plas 
- [An Introduction to Statistical Learning](https://www.statlearning.com/), by James, Witten, Hastie, Tibshirani: statistical theory
- [Kaggle](http://kaggle.com):
    - introduction materials
    - participating in challenges way to learn

???

If you want to study machine learning further, there are many excellent
resources.

First the book "Introduction Machine Learning with Python" by scikit-learn core developer Andreas C. Müller and Sarah Guido goes deeper with scikit-learn
 
Then, the Python Data Science Handbook by Jake van der Plas gives a
broader picture of data science beyond machine learning

The book An Introduction to Statistical Learning, by James, Witten,
Hastie, Tibshirani, gives a view on the statistical theory behind the
concepts that we have explored.

And finally, Kaggle has good introduction materials and participating in
challenges is a great way to learn.

---

# Topics we have not covered

- Unsupervised learning
    - Finding order and structure in the data, for instance to group samples, or to transform features
    - Particularly useful because it does not need labels
    - But given labels, supervised learning not unsupervised learning, is more likely to recover the link
- Model inspection
    - Understanding what drives a prediction
    - Useful for debuging, for reasonning
    - Requires a lot of nuance
- Deep learning
    - For images, text, voice: use pretrained models
    - Comes with great computational and human costs, as well as large maintenance costs
    - Not in scikit-learn

???

- Unsupervised learning
    - Finding order and structure in the data, for instance to group samples, or to transform features
    - Particularly useful because it does not need labels
    - But given labels, supervised learning not unsupervised learning, is more likely to recover the link
- Model inspection
    - Understanding what drives a prediction
    - Useful for debuging, for reasonning on the system at hand
    - Requires a lot of nuance
- Deep learning
    - For images, text, voice: use pretrained models
    - Comes with great computational and human costs, as well as large maintenance costs
    - Not in scikit-learn

---
class: middle

# Beyond machine-learning: The bigger picture

???

And finally, I'd like to talk about the bigger picture, beyond the
technical aspects of machine learning, how it fits in wider questions.

---

# Validation and evaluation matter

A measure of prediction accuracy is an imperfect estimate of how the model will actually generalize to new data

- As you narrow down on a solution, spend increasingly more effort on validating it
- Many splits in your cross-validation

???

Validation and evaluation are often the weak point of an analysis. They
are key to achieving reliable predictive models.

Even with cross-validation, a measure of prediction accuracy is an imperfect estimate of how the model will actually generalize to new data

- As you narrow down on a solution, spend increasingly more effort on validating it
- Many splits in your cross-validation. This brings computational cost, but if you can't afford to evaluate it, you can't afford to use it or trust it


---
  
# Machine learning as a small part of the problem most of the times. 

- How to approach the full problem (the full value chain)
- Acquiring more/better data is often more important than using fancy models
- Challenges to put it in production
    - Define put in production
    - Technical debt (simpler models are easier to maintain, require less compute power)
    - Drifts of the data distribution (require monitoring)

---

# Technical craft is not all

We gave methodological elements, but these are not enough to always have solid conclusion from a statistical standpoint.


Once you know how to run the software, the biggest challenges are understanding the data, its shortcomings, and what can and cannot be concluded from an analysis
  - Automating machine learning does not solve data science
  - Domain knowledge and critical thinking about the data

---

# How the predictions are used

Errors mean different things
- Operational risk:
    - Add placement: errors are harmless
    - Medicine: errors can kill
- Operational logic: Better a false detection or a miss?
    - Detecting brain tumors:
        - Sent to surgery: false detections are very dangerous
        - Given an MR scan: misses should be avoided

???

First, when designing a machine-learning system, we need to think about
how the predictions are used. Errors mean different things in different
application contexts.

The amount of operational risk changes the deal. When decided which
advertisement should be presented, the error is rather harmless. On the
opposite, in medicine, errors can kill.

What is done with a prediction also matters, in particular it can tell us
if a false detection or a miss is less coslty. For instance, when
detecting brain tumors, if the patient is sent to surgery given a
detection, false detections are very dangerous. On the opposite, if the
patent is a given an MR scan to confirm the prediction, misses should
probably be avoided.

---

# How the predictions are used

The predictions may modify how the system is functions:
  - Predicting who will benefit from an hospital stay may overcrowd some units of the hospital, and thus change the positive impact of hospitals on inpatients

---

# Choice of the output/the labeled dataset

- What we chose to predict is a very loaded choice
- Interesting labels are often hard to get, focusing on the "easy" ways
  of accumulating labels comes with biases
- Our target may be a proxy of the quantity of interest


---

# Biases in the data

- The data may not reflect the ground truth
    - Disease monitoring is function of testing policy
    - It may change with time, it may be uneven across the population (eg higher quality data for rich people)
- The state of affaires may not be the desired one
    - For equal qualifications and responsibilities, women are typically payed less than men. A learner will pick this up and perpetuate inequalities

---

# Prediction models versus causal models

- People that go to the hospital die more than people who do not:
    - Fallacy: comparing different populations
- Having a heart pressure greater than a threshold triggers specific care which is good. A learner will pick up above-threshold heart pressure as good for you
- In a pure predictive settings, these learners are correct to use these informations for their predictions. However 1) they should not be trusted when designing interventions on the systems 2) interpretation is subject to caution

---

# Societal impact

AI systems = loans, jobs, medical treatement, law enforcement

https://fairlearn.org/: intro to some problems 

ML can shift decision logic, power structures, operational costs
  - It induces changes in our society. Let us make it better
  - Challenges at the intersection of technology and society. No solution will be purely technical

???

These challenges with biases in the data, feedback loops of the
predictions, can be very important, because prediction models may affect
people's lives.

Today, AI systems can be used to allocate loans, screen job applicants,
prioritise medical treatement, help law enforcement or court decisions.

If you know scikit-learn, [fairlearn](https://fairlearn.org) is a simple
resource to understand some problems caused by
a too naive application of machine learning methods.

ML or AI can shift decision logic, power structures, operational costs
  - As all technology, it induces changes in our society. Let us think
    about how to make it better, even though this is a difficult question
  - Responsible use of machine learning involves challenges at the intersection of technology and society. No solution will be purely technical

---

.center[
# Your move
]


- Machine learning drives one of the most important technological revolution of our time.
- It is a fantastic opportunity to improve our world
- Scikit-learn: lifting technical roadblocks as much as possible
    - empower people
    - to solve the problems that matters to them

???

- Machine learning drives one of the most important technological revolution of our time.
- It is a fantastic opportunity to improve human condition
- With scikit-learn, and this MOOC, we try to lift as much as possible
  the technical roadblocks, and  wee hope that we can empower a great variety of people, with different mindsets and dreams, to solve the problems that matters to them

Thank you for being part of this adventure!