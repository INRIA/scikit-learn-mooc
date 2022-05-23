
# Concluding remarks

```{admonition} Last lesson!
:class: remove-from-content-only

- We have covered a lot of material so far

- Congratulations for getting there!

- And thank you to everyone involved, the instructors, all the support
  staff, the people who helped on the forum, and you, the students, for
  the hard work!
```

```{admonition} Goals of this lesson
:class: remove-from-content-only

- Summarizing the big messages of the MOOC
- Going further with machine learning
- Bringing value: The bigger picture beyond machine-learning
```

## The big messages of the MOOC


### 1. The machine learning pipeline

- Predictive models are learned on a train set and then applied to new
  data, a "test set"

- Scikit-learn models are built from a data matrix, of a given number of
  features for each observation

- Transformations of the data are often necessary
  - Typically, encoding of the categorical variables
  - They must only use information available at train time
  - For this, use the scikit-learn Pipeline object

### 2. Adapting model complexity to the data

- Models seek to minimize the error on the test set
  - Minimizing error on the train set does not suffice
  - But too large train error can detect underfit: models too simple for
    the data

- Models come with multiple hyper-parameters
  - They can control model complexity
  - Selecting hyper-parameters is important
  - In scikit-learn this is done with objects such as GridSearchCV,
    RandomSearchCV...

### 3. Specific models

- Understanding the models
  - Helps knowing when they are suited to the data 
  - Gives intuitions on how to debug them

- Linear models
  - build predictions by combining the values of features
  - Particularly useful for data with many features or few observations
  - Can benefit from non-linear feature engineering

- Tree-based models:
  - Build predictions by combining a series of binary choices (such as
    thresholds on the values of the various attributes)
  - Particularly suited for tabular data, where columns are quantities
    of different nature, or have missing values
  - HistGradientBoostingRegressor and Classifier are goto methods that
    you are strongly advised to check out


## Going further with machine learning

Let us give a few pointers on going further with machine learning.

### Learning more about scikit-learn

- The [scikit-learn doc](http://scikit-learn.org)
    - The documentation is rich, didactic, continuously improving
- These docs comprise
  - An user guide: Gives the intuition behind every machine-learning
    method, and how it can be useful
  - API docs: Every function, every parameter is explained
  - Examples: each example tries to demonstrate the good use of the software
- Where to ask questions:
    - [Stackoverflow](https://stackoverflow.com/)
    - [GitHub discussions for scikit-learn](https://github.com/scikit-learn/scikit-learn/discussions)

### We are an open-source community

- Free, open, driven by a community, trying to be inclusive
- You can contribute
    - Build a community: helping each other, helping training, communication,
      advocacy
    - Curate information: our developers have information overflow
    - [Contributing code](https://scikit-learn.org/stable/developers/contributing.html#contributing-code) is technical
	- Learn software engineering (if you don't know where to start,
     [Software Carpentry](https://software-carpentry.org/lessons) is a good
     resource)
	- Learn git, github (https://lab.github.com/)

### Topics we have not covered

- Unsupervised learning
    - Finding order and structure in the data, for instance to group samples, or to transform features
    - Particularly useful because it does not need labels
    - But given labels, supervised learning not unsupervised learning, is more likely to recover the link between data and labels

- Model inspection
    - Understanding what drives a prediction
    - Useful for debugging, for reasoning on the system at hand
    - Requires a lot of nuance

- Deep learning
    - Often not better than gradient boosting trees for classification or regression
      on tabular data
    - But more flexible: can work natively with tasks that involve variable length
      structures in the input and output of the model (e.g. speech to text)
    - For images, text, voice: use pretrained models
    - Comes with great computational and human costs, as well as large maintenance costs
    - Not in scikit-learn: have a look at resources on pytorch and tensorflow to get started!

### Studying machine learning further

- [Introduction to Machine Learning with
  Python](https://www.oreilly.com/library/view/introduction-to-machine/9781449369880/)
  by Andreas C. Müller, Sarah Guido: explains more advanced use of
  scikit-learn
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) by Jake van der Plas, a broader picture of data science,
  beyond scikit-learn
- [An Introduction to Statistical Learning](https://www.statlearning.com/),
  by James, Witten, Hastie, Tibshirani: statistical theory 
  behind the concepts that we have explored
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) by Aurélien Géron
- [Kaggle](http://kaggle.com):
    - Good introduction materials
    - Participating in challenges, teaming with others, and reading
      the solutions shared by the winners in the discussions is a
      good way to learn

## Bringing value: The bigger picture beyond machine-learning

We will now touch briefly how machine learning fits in wider questions,
how it may fail, and societal aspects.


### Validation and evaluation matter

Validation and evaluation are often the weak point of an analysis. They
are key to achieving reliable predictive models.

Even with cross-validation, a measure of prediction accuracy is an imperfect estimate of how the model will actually generalize to new data

- As you narrow down on a solution, spend increasingly more effort on validating it
- Many splits in your cross-validation. This brings computational cost, but if you can't afford to evaluate it, you can't afford to use it or trust it

Try to think carefully about ways the training set
might not be completely representatives of the future data the
model will make predictions upon. In particular if the model
makes predictions that affect people's lives, are you sure
the training and evaluation data you collected cover a diverse
enough set of different demographics? What can you do
to increase coverage of diverse groups?

Another way to phrase this recommendation is to try to identify
any sampling bias in the data acquisition process.

### Machine learning is a small part of the problem most of the times 

- How to approach the full problem (the full value chain)
- Acquiring more/better data is often more important than using fancy models
- Putting in production: when the model is used routinely
    - Technical debt (simpler models are easier to maintain, require less compute power)
    - Drifts of the data distribution (requires monitoring)

### Technical craft is not all

We gave methodological elements, but these are not enough to always have
solid conclusion from a statistical standpoint.


Once you know how to run the software, the biggest challenges are understanding the data, its shortcomings, and what can and cannot be concluded from an analysis
  - Automating machine learning does not solve data science
  - Domain knowledge and critical thinking about the data


### How the predictions are used

When designing a machine-learning system, we need to think about
how the predictions are used.

Errors mean different things in different application contexts.

- Operational risk:
    - Advertisement: individual errors can cause wasting a bit of money and annoy people but are otherwise mostly harmless
    - Medicine: errors can kill
- Operational logic: Better a false detection or a miss?
    - Detecting brain tumors:
        - If a patient is sent to surgery: false detections are very dangerous
        - If a patient given an MR scan to confirm the detection:
	  misses should be avoided, as an MR scan is harmless, but
	  missing a person with a brain tumor may delay life-saving treatment

The predictions may modify how the system functions:
  - Predicting who will benefit from a hospital stay may overcrowd some units of the hospital, and thus change the positive impact of hospitals on inpatients

### Choice of the output/the labeled dataset

- What we chose to predict is a very loaded choice
- Interesting labels are often hard to get, focusing on the "easy" ways
  of accumulating labels comes with biases
- Our target may be a proxy of the quantity of interest


### Biases in the data

All data come with biases.

- The data may not reflect the ground truth
    - Disease monitoring is function of testing policy
    - It may change with time, it may be uneven across the population (eg higher quality data for rich people)
- The state of affairs may not be the desired one
    - For equal qualifications and responsibilities, women are typically payed less than men. A machine learning model will pick this up and amplify inequalities

### Prediction models versus causal models

Machine learning models are not driven by causal mechanisms.

- For example people that go to the hospital die more than people who do not:
    - Naive data analysis might conclude that hospitals are bad for health
    - The fallacy is that we are comparing different populations: people
      who go to the hospital typically have a worse baseline health than
      people who do not.
- Another example: having a heart pressure greater than a threshold
  may trigger specific care which is good. An automated learner will pick up
  above-threshold heart pressure as predictor of a health improvement
- In pure predictive settings, these informations are beneficial for the
  predictions. However:
 - they should not be trusted when designing interventions
 - predictive models built on such non-causal information may be brittle
   to changes of operational procedures
 - interpretation is subject to caution


### Societal impact

These challenges with biases in the data, feedback loops of the
predictions, can be very important, because **prediction models may affect
people's lives**.

Today, AI systems are sometimes used to allocate loans, screen job applicants,
prioritise medical treatement, help law enforcement or court decisions.

If you know scikit-learn, [fairlearn](https://fairlearn.org) is a simple
resource to help you understand and assess some
problems caused by a too naive application of machine learning
methods.

ML or AI can shift decision logic, power structures, operational costs
  - As all technology, it induces changes in our society. Let us think
    about how to make it better, even though this is a difficult question
  - Responsible use of machine learning involves challenges at the
    intersection of technology and society. No solution will be
    purely technical

A good discussion on these topic can be found in the short article:
[Medicine’s Machine Learning Problem](https://bostonreview.net/articles/rachel-thomas-medicines-machine-learning-problem).


----

**Your move: choose what you will do with machine learning**


- Machine learning drives one of the most important technological revolutions of our time.
- It is a fantastic opportunity to improve human condition
- With scikit-learn, and this MOOC, we try to lift as many technical roadblocks
  as possible, and  we hope that we can empower a great variety of people, with different mindsets and dreams, to solve the problems that matter to them

Thank you for being part of this adventure!
