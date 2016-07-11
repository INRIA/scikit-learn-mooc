from sklearn.metrics import explained_variance_score, mean_squared_error
from sklearn.cross_validation import train_test_split

def plot_learning_curve(model, err_func=explained_variance_score, N=300, n_runs=10, n_sizes=50, ylim=None):
    sizes = np.linspace(5, N, n_sizes).astype(int)
    train_err = np.zeros((n_runs, n_sizes))
    validation_err = np.zeros((n_runs, n_sizes))
    for i in range(n_runs):
        for j, size in enumerate(sizes):
            xtrain, xtest, ytrain, ytest = train_test_split(
                X, y, train_size=size, random_state=i)
            # Train on only the first `size` points
            model.fit(xtrain, ytrain)
            validation_err[i, j] = err_func(ytest, model.predict(xtest))
            train_err[i, j] = err_func(ytrain, model.predict(xtrain))

    plt.plot(sizes, validation_err.mean(axis=0), lw=2, label='validation')
    plt.plot(sizes, train_err.mean(axis=0), lw=2, label='training')

    plt.xlabel('traning set size')
    plt.ylabel(err_func.__name__.replace('_', ' '))
    
    plt.grid(True)
    
    plt.legend(loc=0)
    
    plt.xlim(0, N-1)
    
    if ylim:
        plt.ylim(ylim)


plt.figure(figsize=(10, 8))
for i, model in enumerate([Lasso(0.01), Ridge(0.06)]):
    plt.subplot(221 + i)
    plot_learning_curve(model, ylim=(0, 1))
    plt.title(model.__class__.__name__)
    
    plt.subplot(223 + i)
    plot_learning_curve(model, err_func=mean_squared_error, ylim=(0, 8000))
