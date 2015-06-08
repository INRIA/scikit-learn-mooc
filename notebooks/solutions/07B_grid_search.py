np.random.seed(42)
for model in [DecisionTreeRegressor(),
              GradientBoostingRegressor(),
              RandomForestRegressor()]:
    parameters = {'max_depth':[3, 5, 7, 9, 11]}

    # Warning: be sure your data is shuffled before using GridSearch!
    clf_grid = grid_search.GridSearchCV(model, parameters)
    clf_grid.fit(X, y_noisy)
    print '------------------------'
    print model.__class__.__name__
    print clf_grid.best_params_
    print clf_grid.best_score_
