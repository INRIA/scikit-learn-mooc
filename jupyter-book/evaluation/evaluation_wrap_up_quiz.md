# 🏁 Wrap-up quiz

**This quiz requires some programming to be answered.**

Open the dataset `bike_rides.csv` with the following commands:

```py
import pandas as pd

cycling = pd.read_csv("../datasets/bike_rides.csv", index_col=0,
                      parse_dates=True)
cycling.index.name = ""
target_name = "power"
data, target = cycling.drop(columns=target_name), cycling[target_name]
data.head()
```

A thorough discussion regarding this dataset is given in the annex. We can
recall what is the data science problem that one wants to solve using this
dataset. The idea is to use measurements from cheap sensors (GPS, heart-rate
monitor, etc.) in order to predict a cyclist power. Power can indeed be
recorded via a cycling power meter device, but this device is rather expensive.

We want to develop a predictive model to predict power from other data.
However, instead of using blindly machine learning, we will introduce some
flavor of classic mechanics: the Newton's second law.

$P_{meca} = (\frac{1}{2} \rho . SC_x . V_{a}^{2} + C_r . mg . \cos \alpha + mg . \sin \alpha + ma) V_d$

where $\rho$ is the air density in kg.m$^{-3}$, $S$ is frontal surface of the
cyclist in m$^{2}$, $C_x$ is the drag coefficient, $V_a$ is the air speed in
m.s$^{-1}$, $C_r$ is the rolling coefficient, $m$ is the mass of the rider and
bicycle in kg, $g$ is the gravitational constant which is equal to 9.81
m.s$^{-2}$, $\alpha$ is the slope in radian, $V_d$ is the rider speed in
m.s$^{-1}$, and $a$ is the rider acceleration in m.s$^{-2}$.

This equation might look a bit complex at first but we can explain with words
what are the different terms within the parenthesis. The first term is the
power that a cyclist is required to produce to fight wind. The second term is
the power that a cyclist is required to produce to fight the rolling resistance
created by the tires on the floor. Then, the third term is the power that a
cyclist is required to produce to go up a hill (if the slope is positive; if
the slope is negative the cyclist does not need to produce any power to go
forward). Finally, the last term is the power that a cyclist is required to
change is speed (i.e. acceleration).

We can simplify the model above by using the data that we have at hand. It
would look like the following.

$P_{meca} = \beta_{1} V_{d}^{3} + \beta_{2} V_{d} + \beta_{3} \sin(\alpha) V_{d} + \beta_{4} a V_{d}$

This model is closer to what we saw previously: it is a linear model trained
on a non-linear feature transformation. We will build, train and evaluate
such a model as part of this exercise. Thus, you need to:

- create a new data matrix containing the cube of the speed, the speed, the
  speed multiplied by the sine of the angle of the slope, and the speed
  multiplied by the acceleration. To compute the angle of the slope, you need
  to take the arc tangent of the slope (`alpha = np.arctan(slope)`);
- using the new data matrix, create a linear predictive model based on a
  [`sklearn.preprocessing.StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
  and a
  [`sklearn.linear_model.RidgeCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html);
- use a [`sklearn.model_selection.ShuffleSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html)
  cross-validation strategy with only 4 splits (`n_splits=4`) to evaluate the
  statistical performance of the model. Use the mean absolute error (MAE) as a
  statistical performance metric. Also, pass the parameter
  `return_estimator=True` and `return_train_score=True` to answer to the
  subsequent questions. Be aware that the `ShuffleSplit` strategy is a naive
  strategy and we will investigate the consequence of making this choice in the
  subsequent questions.

```{admonition} Question
On average, what is the Mean Absolute Error on the test sets obtained through
cross-validation is closest to:

- a) 20 Watts
- b) 50 Watts
- c) 70 Watts
- d) 90 Watts

_Select a single answer_

Hint: pass `scoring="neg_mean_absolute_error"` to the `cross_validation
function to compute the (negative of) the requested metric.
```

+++

```{admonition} Question
Given the model
$P_{meca} = \beta_{1} V_{d}^{3} + \beta_{2} V_{d} + \beta_{3} \sin(\alpha) V_{d} + \beta_{4} a V_{d}$
that you programmed, inspect the weights of the linear models fitted during
cross-validation and select the right affirmations.

- a) $\beta_{1} < \beta_{2} < \beta_{3}$
- b) $\beta_{3} < \beta_{1} < \beta_{2}$
- c) $\beta_{2} < \beta_{3} < \beta_{1}$
- d) $\beta_{1} < 0$
- e) $\beta_{2} < 0$
- f) $\beta_{3} < 0$
- g) $\beta_{4} < 0$
- h) All $\beta$s are $> 0$

_Select several answers_
```

+++

Now, we will create a predictive model that uses all available sensor
measurements such as cadence and heart-rate. Also, we will use a non-linear
regressor, a
[`sklearn.ensemble.HistGradientBoostingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).
Fix the number of maximum iterations to 1000 (`max_iter=1_000`) and activate
the early stopping (`early_stopping=True`). Repeat the previous evaluation
using this regressor.

```{admonition} Question
On average, what is the Mean Absolute Error on the test sets obtained through
cross-validation is closest to:

- a) 20 Watts
- b) 40 Watts
- c) 60 Watts
- d) 80 Watts

_Select a single answer_
```

+++

```{admonition} Question
Comparing both the linear model and the histogram gradient boosting model and
taking into consideration the train and test MAE obtained via cross-validation,
select the right affirmations:

- a) the statistical performance of the histogram gradient-boosting model is
  limited by its underfitting
- b) the statistical performance of the histogram gradient-boosting model is
  limited by its overfitting
- c) the statistical performance of the linear model is limited by its
  underfitting
- d) the statistical performance of the linear model is limited by its
  overfitting

_Select several answers_

Hint: look at the values of the train_score and the test_score collected in the
dictionaries returned by the cross_validate function.
```

+++

In the previous cross-validation, we made the choice of using a `ShuffleSplit`
cross-validation strategy. It means that randomly selected samples were
selected as a testing test ignoring any time dependency between the lines of
the dataframe.

We would like to have a cross-validation strategy that evaluates the capacity
of our model to predict on a completely new bike ride: the samples in the
validation set should only come from rides not present in the training set.

```{admonition} Question
How many bike rides are stored in the dataframe `data`? Do not hesitate to
look at the hints.

- a) 2
- b) 3
- c) 4
- d) 5

_Select a single answer_

Hint: You can check the unique day in the `DatetimeIndex` (the index of the
dataframe `data`). Indeed, we assume that on a given day the rider went cycling
at most once per day.
Hint: You can access to the date and time of a `DatetimeIndex` using
`df.index.date` and `df.index.time`, respectively.
```

+++

Instead of using the naive `ShuffleSplit` strategy, we will use a strategy that
takes into account the group defined by each individual date. It
corresponds to a bike ride. We would like to have a cross-validation strategy
that should evaluate the capacity of our model to predict on a completely
new bike ride. Therefore, the test samples should be independent from other
rides. Therefore, we can use a `LeaveOneGroupOut` strategy: at each iteration
of the cross-validation, we will keep a bike ride for the evaluation and use
all other bike rides to train our model.

Thus, you concretely need to:

- create a variable called `group` that is a 1D numpy arrray containing the
  index of each ride present in the dataframe. Therefore, the length of `group`
  will be equal to the number of samples in `data`. If we were having 2 bike
  rides, we would expect the indices 0 and 1 in `group` to differentiate the
  bike ride. You can use
  [`pd.factorize`](https://pandas.pydata.org/docs/reference/api/pandas.factorize.html)
  to encode any Python types into integer indices.
- create a cross-validation object named `cv` using the
  [`sklearn.model_selection.LeaveOneGroupOut`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html#sklearn.model_selection.LeaveOneGroupOut)
  strategy.
- evaluate both the linear and histogram gradient boosting models with this
  strategy.

```{admonition} Question
Using the previous evaluations (with both `ShuffleSplit` and
`LeaveOneGroupOut`) and looking at the train and test errors for both models,
select the right affirmations:

- a) the statistical performance of the histogram gradient-boosting model is
  limited by its underfitting
- b) the statistical performance of the histogram gradient-boosting model is
  limited by its overfitting
- c) the statistical performance of the linear model is limited by its
  underfitting
- d) the statistical performance of the linear model is limited by its
  overfitting
- e) `ShuffleSplit` is giving over-optimistic results for the linear model
- f) `LeaveOneGroupOut` is giving over-optimistic results for the linear model
- g) both cross-validation strategies are equivalent for the linear model
- h) `ShuffleSplit` is giving over-optimistic results for the histogram
  gradient boosting
- i) `LeaveOneGroupOut` is giving over-optimistic results for the histogram
  gradient boosting
- j) both cross-validation strategies are equivalent for the histogram
  gradient-boosting
- k) in general, the standard deviation of the train and test errors increased
  using the `LeaveOneGroupOut` cross-validation
- l) in general, the standard deviation of the train and test errors decreased
  using the `LeaveOneGroupOut` cross-validation

_Select several answers_
```

+++

Now, we will go in details by picking a single ride for the testing and analyse
the predictions of the models for this test ride. To do so, we can reuse the
`LeaveOneGroupOut` cross-validation object in the following manner:

```py
cv = LeaveOneGroupOut()
train_indices, test_indices = list(cv.split(data, target, groups=groups))[0]
data_train, data_test = data.iloc[train_indices], data.iloc[test_indices]
target_train = target.iloc[train_indices]
target_test = target.iloc[test_indices]
```

Now, fit both the linear model and the histogram gradient boosting regressor
models on the training data and collect the prediction on the testing data.
Make a scatter plot where on the x-axis, you will plot the measured powers
(true target) and on the y-axis, you will plot the predicted powers
(predicted target). Do two separated plots for each model.

```{admonition} Question
By analysing the plots, select the right affirmations:

- a) the linear regressor tends to under-predict samples with high power
- b) the linear regressor tends to over-predict samples with high power
- c) the linear regressor makes catastrophic predictions for samples with low
  power
- d) the histogram gradient boosting regressor tends to under-predict samples
  with high power
- e) the histogram gradient boosting regressor tends to over-predict samples
  with high power
- f) the histogram gradient boosting makes catastrophic predictions for samples
  with low power

_Select several answers_
```

+++

Store in the same pandas dataframe the true target and the predictions of each
model. You can give meaningful column name for each of the target (true and
predicted). Then, select a slice of the data corresponding to a range of a
given hour: from 5.00 pm to 5.05 pm. You can achieve such selection with the
following:

```python
time_slice = slice("2020-08-18 17:00:00", "2020-08-18 17:05:00")
target_dataframe.loc[time_slice, :]
```

Plot the different true targets and predictions to answer to the following
question:

```{admonition} Question
By using the previous plot, select the right affirmations:

- a) the linear model is more accurate than the histogram gradient boosting
  regressor
- b) the histogram gradient boosting regressor is more accurate than the linear
  model
- c) the linear model predicts smoother outputs than the histogram gradient
  boosting regressor
- d) the histogram gradient boosting regressor predicts smoother outputs
  than the linear model

_Select several answers_
```
