from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

boston = load_boston()
text_train, text_test, y_train, y_test = train_test_split(boston.data,
                                                          boston.target,
                                                          test_size=0.25,
                                                          random_state=123)

pipeline = make_pipeline(StandardScaler(),
                         PolynomialFeatures(),
                         Ridge())

grid = GridSearchCV(pipeline,
                    param_grid={'polynomialfeatures__degree': [1, 2, 3]}, cv=5)

grid.fit(text_train, y_train)

print('best parameters:', grid.best_params_)
print('best score:', grid.best_score_)
print('test score:', grid.score(text_test, y_test))
