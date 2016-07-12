cv = KFold(n=len(iris.target), n_folds=3)
cross_val_score(classifier, iris.data, iris.target, cv=cv)
