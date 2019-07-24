# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,markdown_files//md,python_scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] {"deletable": true, "editable": true}
# # Out-of-core Learning - Large Scale Text Classification for Sentiment Analysis

# %% [markdown] {"deletable": true, "editable": true}
# ## Scalability Issues

# %% [markdown] {"deletable": true, "editable": true}
# The `sklearn.feature_extraction.text.CountVectorizer` and `sklearn.feature_extraction.text.TfidfVectorizer` classes suffer from a number of scalability issues that all stem from the internal usage of the `vocabulary_` attribute (a Python dictionary) used to map the unicode string feature names to the integer feature indices.
#
# The main scalability issues are:
#
# - **Memory usage of the text vectorizer**: all the string representations of the features are loaded in memory
# - **Parallelization problems for text feature extraction**: the `vocabulary_` would be a shared state: complex synchronization and overhead
# - **Impossibility to do online or out-of-core / streaming learning**: the `vocabulary_` needs to be learned from the data: its size cannot be known before making one pass over the full dataset
#     
#     
# To better understand the issue let's have a look at how the `vocabulary_` attribute work. At `fit` time the tokens of the corpus are uniquely indentified by a integer index and this mapping stored in the vocabulary:

# %% {"deletable": true, "editable": true}
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df=1)

vectorizer.fit([
    "The cat sat on the mat.",
])
vectorizer.vocabulary_

# %% [markdown] {"deletable": true, "editable": true}
# The vocabulary is used at `transform` time to build the occurrence matrix:

# %% {"deletable": true, "editable": true}
X = vectorizer.transform([
    "The cat sat on the mat.",
    "This cat is a nice cat.",
]).toarray()

print(len(vectorizer.vocabulary_))
print(vectorizer.get_feature_names())
print(X)

# %% [markdown] {"deletable": true, "editable": true}
# Let's refit with a slightly larger corpus:

# %% {"deletable": true, "editable": true}
vectorizer = CountVectorizer(min_df=1)

vectorizer.fit([
    "The cat sat on the mat.",
    "The quick brown fox jumps over the lazy dog.",
])
vectorizer.vocabulary_

# %% [markdown] {"deletable": true, "editable": true}
# The `vocabulary_` is the (logarithmically) growing with the size of the training corpus. Note that we could not have built the vocabularies in parallel on the 2 text documents as they share some words hence would require some kind of shared datastructure or synchronization barrier which is complicated to setup, especially if we want to distribute the processing on a cluster.
#
# With this new vocabulary, the dimensionality of the output space is now larger:

# %% {"deletable": true, "editable": true}
X = vectorizer.transform([
    "The cat sat on the mat.",
    "This cat is a nice cat.",
]).toarray()

print(len(vectorizer.vocabulary_))
print(vectorizer.get_feature_names())
print(X)

# %% [markdown] {"deletable": true, "editable": true}
# ## The IMDb movie dataset

# %% [markdown] {"deletable": true, "editable": true}
# To illustrate the scalability issues of the vocabulary-based vectorizers, let's load a more realistic dataset for a classical text classification task: sentiment analysis on text documents. The goal is to tell apart negative from positive movie reviews from the [Internet Movie Database](http://www.imdb.com) (IMDb).
#
# In the following sections, with a [large subset](http://ai.stanford.edu/~amaas/data/sentiment/) of movie reviews from the IMDb that has been collected by Maas et al. 
#
# - A. L. Maas, R. E. Daly, P. T. Pham, D. Huang, A. Y. Ng, and C. Potts. Learning Word Vectors for Sentiment Analysis. In the proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pages 142–150, Portland, Oregon, USA, June 2011. Association for Computational Linguistics. 
#
# This dataset contains 50,000 movie reviews, which were split into 25,000 training samples and 25,000 test samples. The reviews are labeled as either negative (neg) or positive (pos). Moreover, *positive* means that a movie received >6 stars on IMDb; negative means that a movie received <5 stars, respectively.
#
#
# Assuming that the `../fetch_data.py` script was run successfully the following files should be available:

# %% {"deletable": true, "editable": true}
import os

train_path = os.path.join('datasets', 'IMDb', 'aclImdb', 'train')
test_path = os.path.join('datasets', 'IMDb', 'aclImdb', 'test')

# %% [markdown] {"deletable": true, "editable": true}
# Now, let's load them into our active session via scikit-learn's `load_files` function

# %% {"deletable": true, "editable": true}
from sklearn.datasets import load_files

train = load_files(container_path=(train_path),
                   categories=['pos', 'neg'])

test = load_files(container_path=(test_path),
                  categories=['pos', 'neg'])

# %% [markdown] {"deletable": true, "editable": true}
# <div class="alert alert-warning">
#     <b>NOTE</b>:
#      <ul>
#       <li>
#       Since the movie datasets consists of 50,000 individual text files, executing the code snippet above may take ~20 sec or longer.
#       </li>
#     </ul>
# </div>

# %% [markdown] {"deletable": true, "editable": true}
# The `load_files` function loaded the datasets into `sklearn.datasets.base.Bunch` objects, which are Python dictionaries:

# %% {"deletable": true, "editable": true}
train.keys()

# %% [markdown] {"deletable": true, "editable": true}
# In particular, we are only interested in the `data` and `target` arrays.

# %% {"deletable": true, "editable": true}
import numpy as np

for label, data in zip(('TRAINING', 'TEST'), (train, test)):
    print('\n\n%s' % label)
    print('Number of documents:', len(data['data']))
    print('\n1st document:\n', data['data'][0])
    print('\n1st label:', data['target'][0])
    print('\nClass names:', data['target_names'])
    print('Class count:', 
          np.unique(data['target']), ' -> ',
          np.bincount(data['target']))

# %% [markdown] {"deletable": true, "editable": true}
# As we can see above the `'target'` array consists of integers `0` and `1`, where `0` stands for negative and `1` stands for positive.

# %% [markdown] {"deletable": true, "editable": true}
# ## The Hashing Trick

# %% [markdown] {"deletable": true, "editable": true}
# Remember the bag of word representation using a vocabulary based vectorizer:
#
# <img src="figures/bag_of_words.svg" width="100%">

# %% [markdown] {"deletable": true, "editable": true}
# To workaround the limitations of the vocabulary-based vectorizers, one can use the hashing trick. Instead of building and storing an explicit mapping from the feature names to the feature indices in a Python dict, we can just use a hash function and a modulus operation:

# %% [markdown] {"deletable": true, "editable": true}
# <img src="figures/hashing_vectorizer.svg" width="100%">

# %% [markdown] {"deletable": true, "editable": true}
# More info and reference for the original papers on the Hashing Trick in the [following site](http://www.hunch.net/~jl/projects/hash_reps/index.html) as well as a description specific to language [here](http://blog.someben.com/2013/01/hashing-lang/).

# %% {"deletable": true, "editable": true}
from sklearn.utils.murmurhash import murmurhash3_bytes_u32

# encode for python 3 compatibility
for word in "the cat sat on the mat".encode("utf-8").split():
    print("{0} => {1}".format(
        word, murmurhash3_bytes_u32(word, 0) % 2 ** 20))

# %% [markdown] {"deletable": true, "editable": true}
# This mapping is completely stateless and the dimensionality of the output space is explicitly fixed in advance (here we use a modulo `2 ** 20` which means roughly 1M dimensions). The makes it possible to workaround the limitations of the vocabulary based vectorizer both for parallelizability and online / out-of-core learning.

# %% [markdown] {"deletable": true, "editable": true}
# The `HashingVectorizer` class is an alternative to the `CountVectorizer` (or `TfidfVectorizer` class with `use_idf=False`) that internally uses the murmurhash hash function:

# %% {"deletable": true, "editable": true}
from sklearn.feature_extraction.text import HashingVectorizer

h_vectorizer = HashingVectorizer(encoding='latin-1')
h_vectorizer

# %% [markdown] {"deletable": true, "editable": true}
# It shares the same "preprocessor", "tokenizer" and "analyzer" infrastructure:

# %% {"deletable": true, "editable": true}
analyzer = h_vectorizer.build_analyzer()
analyzer('This is a test sentence.')

# %% [markdown] {"deletable": true, "editable": true}
# We can vectorize our datasets into a scipy sparse matrix exactly as we would have done with the `CountVectorizer` or `TfidfVectorizer`, except that we can directly call the `transform` method: there is no need to `fit` as `HashingVectorizer` is a stateless transformer:

# %% {"deletable": true, "editable": true}
docs_train, y_train = train['data'], train['target']
docs_valid, y_valid = test['data'][:12500], test['target'][:12500]
docs_test, y_test = test['data'][12500:], test['target'][12500:]

# %% [markdown] {"deletable": true, "editable": true}
# The dimension of the output is fixed ahead of time to `n_features=2 ** 20` by default (nearly 1M features) to minimize the rate of collision on most classification problem while having reasonably sized linear models (1M weights in the `coef_` attribute):

# %% {"deletable": true, "editable": true}
h_vectorizer.transform(docs_train)

# %% [markdown] {"deletable": true, "editable": true}
# Now, let's compare the computational efficiency of the `HashingVectorizer` to the `CountVectorizer`:

# %% {"deletable": true, "editable": true}
h_vec = HashingVectorizer(encoding='latin-1')
# %timeit -n 1 -r 3 h_vec.fit(docs_train, y_train)

# %% {"deletable": true, "editable": true}
count_vec =  CountVectorizer(encoding='latin-1')
# %timeit -n 1 -r 3 count_vec.fit(docs_train, y_train)

# %% [markdown] {"deletable": true, "editable": true}
# As we can see, the HashingVectorizer is much faster than the Countvectorizer in this case.

# %% [markdown] {"deletable": true, "editable": true}
# Finally, let us train a LogisticRegression classifier on the IMDb training subset:

# %% {"deletable": true, "editable": true}
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

h_pipeline = Pipeline([
    ('vec', HashingVectorizer(encoding='latin-1')),
    ('clf', LogisticRegression(random_state=1)),
])

h_pipeline.fit(docs_train, y_train)

# %% {"deletable": true, "editable": true}
print('Train accuracy', h_pipeline.score(docs_train, y_train))
print('Validation accuracy', h_pipeline.score(docs_valid, y_valid))

# %% {"deletable": true, "editable": true}
import gc

del count_vec
del h_pipeline

gc.collect()

# %% [markdown] {"deletable": true, "editable": true}
# # Out-of-Core learning

# %% [markdown] {"deletable": true, "editable": true}
# Out-of-Core learning is the task of training a machine learning model on a dataset that does not fit into memory or RAM. This requires the following conditions:
#     
# - a **feature extraction** layer with **fixed output dimensionality**
# - knowing the list of all classes in advance (in this case we only have positive and negative reviews)
# - a machine learning **algorithm that supports incremental learning** (the `partial_fit` method in scikit-learn).
#
# In the following sections, we will set up a simple batch-training function to train an `SGDClassifier` iteratively. 

# %% [markdown] {"deletable": true, "editable": true}
# But first, let us load the file names into a Python list:

# %% {"deletable": true, "editable": true}
train_path = os.path.join('datasets', 'IMDb', 'aclImdb', 'train')
train_pos = os.path.join(train_path, 'pos')
train_neg = os.path.join(train_path, 'neg')

fnames = [os.path.join(train_pos, f) for f in os.listdir(train_pos)] +\
         [os.path.join(train_neg, f) for f in os.listdir(train_neg)]

fnames[:3]

# %% [markdown] {"deletable": true, "editable": true}
# Next, let us create the target label array:

# %% {"deletable": true, "editable": true}
y_train = np.zeros((len(fnames), ), dtype=int)
y_train[:12500] = 1
np.bincount(y_train)

# %% [markdown] {"deletable": true, "editable": true}
# Now, we implement the `batch_train function` as follows:

# %% {"deletable": true, "editable": true}
from sklearn.base import clone

def batch_train(clf, fnames, labels, iterations=25, batchsize=1000, random_seed=1):
    vec = HashingVectorizer(encoding='latin-1')
    idx = np.arange(labels.shape[0])
    c_clf = clone(clf)
    rng = np.random.RandomState(seed=random_seed)
    
    for i in range(iterations):
        rnd_idx = rng.choice(idx, size=batchsize)
        documents = []
        for i in rnd_idx:
            with open(fnames[i], 'r', encoding='latin-1') as f:
                documents.append(f.read())
        X_batch = vec.transform(documents)
        batch_labels = labels[rnd_idx]
        c_clf.partial_fit(X=X_batch, 
                          y=batch_labels, 
                          classes=[0, 1])
      
    return c_clf


# %% [markdown] {"deletable": true, "editable": true}
# Note that we are not using `LogisticRegression` as in the previous section, but we will use a `SGDClassifier` with a logistic cost function instead. SGD stands for `stochastic gradient descent`, an optimization alrogithm that optimizes the weight coefficients iteratively sample by sample, which allows us to feed the data to the classifier chunk by chuck.

# %% [markdown] {"deletable": true, "editable": true}
# And we train the `SGDClassifier`; using the default settings of the `batch_train` function, it will train the classifier on 25*1000=25000 documents. (Depending on your machine, this may take >2 min)

# %% {"deletable": true, "editable": true}
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(loss='log', random_state=1, max_iter=1000)

sgd = batch_train(clf=sgd,
                  fnames=fnames,
                  labels=y_train)

# %% [markdown] {"deletable": true, "editable": true}
# Eventually, let us evaluate its performance:

# %% {"deletable": true, "editable": true}
vec = HashingVectorizer(encoding='latin-1')
sgd.score(vec.transform(docs_test), y_test)

# %% [markdown] {"deletable": true, "editable": true}
# ### Limitations of the Hashing Vectorizer

# %% [markdown] {"deletable": true, "editable": true}
# Using the Hashing Vectorizer makes it possible to implement streaming and parallel text classification but can also introduce some issues:
#     
# - The collisions can introduce too much noise in the data and degrade prediction quality,
# - The `HashingVectorizer` does not provide "Inverse Document Frequency" reweighting (lack of a `use_idf=True` option).
# - There is no easy way to inverse the mapping and find the feature names from the feature index.
#
# The collision issues can be controlled by increasing the `n_features` parameters.
#
# The IDF weighting might be reintroduced by appending a `TfidfTransformer` instance on the output of the vectorizer. However computing the `idf_` statistic used for the feature reweighting will require to do at least one additional pass over the training set before being able to start training the classifier: this breaks the online learning scheme.
#
# The lack of inverse mapping (the `get_feature_names()` method of `TfidfVectorizer`) is even harder to workaround. That would require extending the `HashingVectorizer` class to add a "trace" mode to record the mapping of the most important features to provide statistical debugging information.
#
# In the mean time to debug feature extraction issues, it is recommended to use `TfidfVectorizer(use_idf=False)` on a small-ish subset of the dataset to simulate a `HashingVectorizer()` instance that have the `get_feature_names()` method and no collision issues.

# %% [markdown] {"deletable": true, "editable": true}
# <div class="alert alert-success">
#     <b>EXERCISE</b>:
#      <ul>
#       <li>
#       In our implementation of the batch_train function above, we randomly draw *k* training samples as a batch in each iteration, which can be considered as a random subsampling ***with*** replacement. Can you modify the `batch_train` function so that it iterates over the documents ***without*** replacement, i.e., that it uses each document ***exactly once*** per iteration?
#       </li>
#     </ul>
# </div>

# %% {"deletable": true, "editable": true}
# # %load solutions/23_batchtrain.py
