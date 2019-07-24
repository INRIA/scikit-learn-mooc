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

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# # Case Study - Text classification for SMS spam detection

# %% [markdown]
# We first load the text data from the `dataset` directory that should be located in your notebooks directory, which we created by running the `fetch_data.py` script from the top level of the GitHub repository.
#
# Furthermore, we perform some simple preprocessing and split the data array into two parts:
#
# 1. `text`: A list of lists, where each sublists contains the contents of our emails
# 2. `y`: our SPAM vs HAM labels stored in binary; a 1 represents a spam message, and a 0 represnts a ham (non-spam) message. 

# %%
import os

with open(os.path.join("datasets", "smsspam", "SMSSpamCollection")) as f:
    lines = [line.strip().split("\t") for line in f.readlines()]

text = [x[1] for x in lines]
y = [int(x[0] == "spam") for x in lines]

# %%
text[:10]

# %%
y[:10]

# %%
print('Number of ham and spam messages:', np.bincount(y))

# %%
type(text)

# %%
type(y)

# %% [markdown]
# Next, we split our dataset into 2 parts, the test and training dataset:

# %%
from sklearn.model_selection import train_test_split

text_train, text_test, y_train, y_test = train_test_split(text, y, 
                                                          random_state=42,
                                                          test_size=0.25,
                                                          stratify=y)

# %% [markdown]
# Now, we use the CountVectorizer to parse the text data into a bag-of-words model.

# %%
from sklearn.feature_extraction.text import CountVectorizer

print('CountVectorizer defaults')
CountVectorizer()

# %%
vectorizer = CountVectorizer()
vectorizer.fit(text_train)

X_train = vectorizer.transform(text_train)
X_test = vectorizer.transform(text_test)

# %%
print(len(vectorizer.vocabulary_))

# %%
X_train.shape

# %%
print(vectorizer.get_feature_names()[:20])

# %%
print(vectorizer.get_feature_names()[2000:2020])

# %%
print(X_train.shape)
print(X_test.shape)

# %% [markdown]
# ### Training a Classifier on Text Features

# %% [markdown]
# We can now train a classifier, for instance a logistic regression classifier, which is a fast baseline for text classification tasks:

# %%
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf

# %%
clf.fit(X_train, y_train)

# %% [markdown]
# We can now evaluate the classifier on the testing set. Let's first use the built-in score function, which is the rate of correct classification in the test set:

# %%
clf.score(X_test, y_test)

# %% [markdown]
# We can also compute the score on the training set to see how well we do there:

# %%
clf.score(X_train, y_train)


# %% [markdown]
# # Visualizing important features

# %%
def visualize_coefficients(classifier, feature_names, n_top_features=25):
    # get coefficients with large absolute values 
    coef = classifier.coef_.ravel()
    positive_coefficients = np.argsort(coef)[-n_top_features:]
    negative_coefficients = np.argsort(coef)[:n_top_features]
    interesting_coefficients = np.hstack([negative_coefficients, positive_coefficients])
    # plot them
    plt.figure(figsize=(15, 5))
    colors = ["tab:orange" if c < 0 else "tab:blue" for c in coef[interesting_coefficients]]
    plt.bar(np.arange(2 * n_top_features), coef[interesting_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 2 * n_top_features + 1), feature_names[interesting_coefficients], rotation=60, ha="right");


# %%
visualize_coefficients(clf, vectorizer.get_feature_names())

# %%
vectorizer = CountVectorizer(min_df=2)
vectorizer.fit(text_train)

X_train = vectorizer.transform(text_train)
X_test = vectorizer.transform(text_test)

clf = LogisticRegression()
clf.fit(X_train, y_train)

print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))

# %%
len(vectorizer.get_feature_names())

# %%
print(vectorizer.get_feature_names()[:20])

# %%
visualize_coefficients(clf, vectorizer.get_feature_names())

# %% [markdown]
# <img src="figures/supervised_scikit_learn.png" width="100%">

# %% [markdown]
# <div class="alert alert-success">
#     <b>EXERCISE</b>:
#      <ul>
#       <li>
#       Use TfidfVectorizer instead of CountVectorizer. Are the results better? How are the coefficients different?
#       </li>
#       <li>
#       Change the parameters min_df and ngram_range of the TfidfVectorizer and CountVectorizer. How does that change the important features?
#       </li>
#     </ul>
# </div>

# %% {"deletable": true, "editable": true}
# # %load solutions/12A_tfidf.py

# %% {"deletable": true, "editable": true}
# # %load solutions/12B_vectorizer_params.py
