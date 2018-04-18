import numpy as np
from sklearn.model_selection import StratifiedKFold


def k_split(x, y, nr_folds, random_state):  # todo test
    kf = StratifiedKFold(n_splits=nr_folds, random_state=random_state, shuffle=True)
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        yield x_train, x_test, y_train, y_test


def get_x_y_from_list_of_tuples(dataset):
    x, y = zip(*dataset)
    x = np.array(x)
    y = np.array(y)
    return x, y
