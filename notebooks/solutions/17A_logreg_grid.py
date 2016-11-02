from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression

digits = load_digits()
X_digits, y_digits = digits.data, digits.target
X_digits_train, X_digits_test, y_digits_train, y_digits_test = train_test_split(X_digits, y_digits, random_state=1)

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}

grid = GridSearchCV(LogisticRegression(), param_grid=param_grid, cv=5, verbose=3)
grid.fit(X_digits_train, y_digits_train)
print('Best score for LogisticRegression: {}'.format(grid.score(X_digits_test, y_digits_test)))
print('Best parameters for LogisticRegression: {}'.format(grid.best_params_))
