from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import numpy as np

rng = np.random.RandomState(1)
X = rng.randint(0, 2, (200, 20))
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

fs_univariate = SelectKBest(k=10)
fs_modelbased = SelectFromModel(RandomForestClassifier(n_estimators=100), threshold='median')

fs_univariate.fit(X, y)
print('Features selected by univariate selection:')
print(fs_univariate.get_support())
print('')

fs_modelbased.fit(X, y)
print('Features selected by model-based selection:')
print(fs_modelbased.get_support())
