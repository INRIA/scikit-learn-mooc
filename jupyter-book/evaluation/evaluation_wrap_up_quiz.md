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
```

A thorough discussion regarding this dataset is given in the annex. We can
recall what is the data science problem that one wants to solve using this
dataset. The idea is to use measurements from cheap sensors (GPS, heart-rate
monitor, etc.) in order to predict a cyclist power. Power can indeed be
recorded via a cycling power meter device, but this device is rather expensive.

We want to develop a predictive model to predict power from other data.
However, instead of using blindly machine learning, we will introduce some
flavor of classic mechanics: the Newton's second law.

$P_{meca} = (\frac{1}{2} \rho . SC_x . V_{a}^{2} + C_r . mg . \cos \alpha + mg . \sin \alpha) V_d$

where $\rho$ is the air density in kg.m$^{-3}$, $S$ is frontal surface of the
cyclist in m$^{2}$, $C_x$ is the drag coefficient, $V_a$ is the air speed in
m.s$^{-1}$, $C_r$ is the rolling coefficient, $m$ is the mass of the rider and
bicycle in kg, $g$ is the gravitational constant which is equal to 9.81
m.s$^{-2}$, $\alpha$ is the slope in radian, and $V_d$ is the rider speed in
m.s$^{-1}$.

This equation might look a bit complex at first but we can explain with words
what are the different terms within the parenthesis. The first term is the
power that a cyclist is required to produce to fight wind. The second term is
the power that a cyclist is required to produce to fight the rolling resistance
created by the tires on the floor. Finally, the last term is the power that a
cyclist is required to produce to go up a hill (if the slope is positive; if
the slope is negative the cyclist do not need to produce any power to go
forward).

We can simplify the model above by using the data that we have at hand. It
would look like the following.

$P_{meca} = \beta_{1} V_{d}^{3} + \beta_{2} V_{d} + \beta_{3} \sin(\alpha) V_{d}$

This model is closer to what we so before: it is a linear model with some
feature expansion. We will create such model and evaluate it. Thus, you need
to:

- create a new data matrix containing the cube of the speed, the speed and the
  speed multiply by the sine of the angle of the slope. The compute the angle
  of the slope, you need to take the arc tangent of the slope
  (`alpha = np.arctan(slope)`);
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
In average, what is the MAE on the test sets obtained through cross-validation:

- a) 20 Watts
- b) 50 Watts
- c) 70 Watts
- d) 90 Watts

_Select a single answer_
```

+++

```{admonition} Question
Given the model
$P_{meca} = \beta_{1} V_{d}^{3} + \beta_{2} V_{d} + \beta_{3} \sin(\alpha) V_{d}$
that you programmed, inspect the weights of the linear models fitted during
cross-validation and select the right affirmations.

- a) $\beta_{1} < \beta_{2} < \beta_{3}$
- b) $\beta_{3} < \beta_{1} < \beta_{2}$
- c) $\beta_{2} < \beta_{3} < \beta_{1}$
- d) $\beta_{1} < 0$
- e) $\beta_{2} < 0$
- f) $\beta_{3} < 0$
- g) All $\beta$s  are $> 0$

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
In average, what is the MAE on the test sets obtained through cross-validation:

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
```

+++

In the previous cross-validation, we made the choice of using a `ShuffleSplit`
cross-validation strategy. It means that randomly selected samples were
selected as a testing test without taking care of the structure of the data.

We will look a bit further regarding the structure of the data and question or
cross-validation choice. If we inspect the dataset `data` index, we can see
that there are some time groups that corresponds to bike rides.

```{admonition} Question
How many bike rides are stored in the dataframe `data`? Do not hesitate to
look at the hints.

- a) 2
- b) 3
- c) 4
- d) 5

_Select a single answer_

Hint: You can check the unique day in the `DatetimeIndex` (the index of the
dataframe `data`). Indeed, we assume that on a given day the rider went a
single time cycling.
Hint: You can access to the date and time of a `DatetimeIndex` using
`df.index.date` and `df.index.time`, respectively.
```

+++

Instead of using the naive `ShuffleSplit` strategy, we will use a strategy that
will take into account the group defined by each individual date. It
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
  bike ride. Be aware that you can use
  [`pd.factorize`](https://pandas.pydata.org/docs/reference/api/pandas.factorize.html)
  to encode any Python types into integer indices.
- create a cross-validation `cv` using the
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
