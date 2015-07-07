import numpy as np
from collections import defaultdict
import os
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_extraction import DictVectorizer

# Can also use pandas!
def process_titanic_line(line):
    # Split line on "," to get fields without comma confusion
    vals = line.strip().split('",')
    # replace spurious " characters
    vals = [v.replace('"', '') for v in vals]
    pclass = int(vals[0])
    survived = int(vals[1])
    name = str(vals[2])
    sex = str(vals[3])
    try:
        age = float(vals[4])
    except ValueError:
        # Blank age
        age = -1
    sibsp = float(vals[5])
    parch = int(vals[6])
    ticket = str(vals[7])
    try:
        fare = float(vals[8])
    except ValueError:
        # Blank fare
        fare = -1
    cabin = str(vals[9])
    embarked = str(vals[10])
    boat = str(vals[11])
    homedest = str(vals[12])
    line_dict = {'pclass': pclass, 'survived': survived, 'name': name, 'sex': sex, 'age': age, 'sibsp': sibsp,
                 'parch': parch, 'ticket': ticket, 'fare': fare, 'cabin': cabin, 'embarked': embarked,
                 'boat': boat, 'homedest': homedest}
    return line_dict


def load_titanic(test_size=.25, feature_skip_tuple=(), random_state=1999):
    f = open(os.path.join('datasets', 'titanic', 'titanic3.csv'))
    # Remove . from home.dest, split on quotes because some fields have commas
    keys = f.readline().strip().replace('.', '').split('","')
    lines = f.readlines()
    f.close()
    string_keys = ['name', 'sex', 'ticket', 'cabin', 'embarked', 'boat',
                   'homedest']
    string_keys = [s for s in string_keys if s not in feature_skip_tuple]
    numeric_keys = ['pclass', 'age', 'sibsp', 'parch', 'fare']
    numeric_keys = [n for n in numeric_keys if n not in feature_skip_tuple]
    train_vectorizer_list = []
    test_vectorizer_list = []

    n_samples = len(lines)
    numeric_data = np.zeros((n_samples, len(numeric_keys)))
    numeric_labels = np.zeros((n_samples,), dtype=int)

    # Doing this twice is horribly inefficient but the file is small...
    for n, l in enumerate(lines):
        line_dict = process_titanic_line(l)
        strings = {k: line_dict[k] for k in string_keys}
        numeric_labels[n] = line_dict["survived"]

    sss = StratifiedShuffleSplit(numeric_labels, n_iter=1, test_size=test_size,
                                 random_state=12)
    # This is a weird way to get the indices but it works
    train_idx = None
    test_idx = None
    for train_idx, test_idx in sss:
        pass

    for n, l in enumerate(lines):
        line_dict = process_titanic_line(l)
        strings = {k: line_dict[k] for k in string_keys}
        if n in train_idx:
            train_vectorizer_list.append(strings)
        else:
            test_vectorizer_list.append(strings)
        numeric_data[n] = np.asarray([line_dict[k]
                                      for k in numeric_keys])

    train_numeric = numeric_data[train_idx]
    test_numeric = numeric_data[test_idx]
    train_labels = numeric_labels[train_idx]
    test_labels = numeric_labels[test_idx]

    vec = DictVectorizer()
    # .toarray() due to returning a scipy sparse array
    train_categorical = vec.fit_transform(train_vectorizer_list).toarray()
    test_categorical = vec.transform(test_vectorizer_list).toarray()
    train_data = np.concatenate([train_numeric, train_categorical], axis=1)
    test_data = np.concatenate([test_numeric, test_categorical], axis=1)
    keys = numeric_keys + string_keys
    return keys, train_data, test_data, train_labels, test_labels


FIELDNAMES = ('polarity', 'id', 'date', 'query', 'author', 'text')

def read_sentiment_csv(csv_file, fieldnames=FIELDNAMES, max_count=None,
             n_partitions=1, partition_id=0):
    import csv  # put the import inside for use in IPython.parallel
    def file_opener(csv_file):
        try:
            open(csv_file, 'r', encoding="latin1").close()
            return open(csv_file, 'r', encoding="latin1")
        except TypeError:
            # Python 2 does not have encoding arg
            return open(csv_file, 'rb')

    texts = []
    targets = []
    with file_opener(csv_file) as f:
        reader = csv.DictReader(f, fieldnames=fieldnames,
                                delimiter=',', quotechar='"')
        pos_count, neg_count = 0, 0
        for i, d in enumerate(reader):
            if i % n_partitions != partition_id:
                # Skip entry if not in the requested partition
                continue

            if d['polarity'] == '4':
                if max_count and pos_count >= max_count / 2:
                    continue
                pos_count += 1
                texts.append(d['text'])
                targets.append(1)

            elif d['polarity'] == '0':
                if max_count and neg_count >= max_count / 2:
                    continue
                neg_count += 1
                texts.append(d['text'])
                targets.append(-1)

    return texts, targets
