def accuracy(true, pred):
    return (true == pred).sum() / float(true.shape[0])


def macro(true, pred):
    scores = []
    for l in np.unique(true):
        scores.append(accuracy(np.where(true != l, 1, 0),
                               np.where(pred != l, 1, 0)))
    return float(sum(scores)) / float(len(scores))

y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2])
y_pred = np.array([0, 1, 1, 0, 1, 1, 2, 2, 2, 2])


print('accuracy:', accuracy(y_true, y_pred))
print('average-per-class accuracy:', macro(y_true, y_pred))
