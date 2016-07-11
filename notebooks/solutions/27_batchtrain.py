import os
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.base import clone
from sklearn.datasets import load_files


def batch_train(clf, fnames, labels, iterations=1,
                batchsize=1000, random_seed=1):
    vec = HashingVectorizer(encoding='latin-1')
    idx = np.arange(labels.shape[0])
    c_clf = clone(clf)
    rng = np.random.RandomState(seed=random_seed)
    shuffled_idx = rng.permutation(range(len(fnames)))
    fnames_ary = np.asarray(fnames)

    for _ in range(iterations):
        for batch in np.split(shuffled_idx, len(fnames) // 1000):
            documents = []
            for fn in fnames_ary[batch]:
                with open(fn, 'r') as f:
                    documents.append(f.read())
            X_batch = vec.transform(documents)
            batch_labels = labels[batch]
            c_clf.partial_fit(X=X_batch,
                              y=batch_labels,
                              classes=[0, 1])

    return c_clf


# Out-of-core Training
train_path = os.path.join('datasets', 'IMDb', 'aclImdb', 'train')
train_pos = os.path.join(train_path, 'pos')
train_neg = os.path.join(train_path, 'neg')

fnames = [os.path.join(train_pos, f) for f in os.listdir(train_pos)] +\
         [os.path.join(train_neg, f) for f in os.listdir(train_neg)]
y_train = np.zeros((len(fnames), ), dtype=int)
y_train[:12500] = 1
np.bincount(y_train)

sgd = SGDClassifier(loss='log', random_state=1)

sgd = batch_train(clf=sgd,
                  fnames=fnames,
                  labels=y_train)


# Testing
test_path = os.path.join('datasets', 'IMDb', 'aclImdb', 'test')
test = load_files(container_path=(test_path),
                  categories=['pos', 'neg'])
docs_test, y_test = test['data'][12500:], test['target'][12500:]

vec = HashingVectorizer(encoding='latin-1')
print('accuracy:', sgd.score(vec.transform(docs_test), y_test))
