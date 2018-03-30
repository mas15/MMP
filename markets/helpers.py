import os
import csv
from random import shuffle
from math import ceil

CORPUS_FILE = os.path.join(os.path.dirname(__file__), "data/sentimental_tweets.csv")


def get_x_y_from_df(df):  # todo to nie tylko tutaj
    target_column = df.columns[-1]
    y = df[target_column].values.ravel()
    x = df.drop(columns=[target_column, "Text"]).values  # todo text jakos ogarnac
    return x, y


def mark_row(row, extr):  # todo mark features on df
    features = extr.extract_features(row['Text'])
    for f, is_in_tweet in features.items():
        if is_in_tweet:
            row[f] = 1
    return row


def mark_features(extr, df):
    for f in extr.features:
        df[f] = 0

    df = df.apply(lambda row: mark_row(row, extr), axis=1)
    return df


def move_column_to_the_end(df, col_name):
    cols = list(df)
    cols.append(cols.pop(cols.index(col_name)))
    df = df.reindex(columns=cols)
    return df


def drop_infrequent_features(df, min_freq=7): # todo set usunac to
    features = df.drop(columns=["Market_change", "Tweet_sentiment", "Text"], errors="ignore")
    cols_with_nr_of_trues = count_nr_of_feature_occurencies(features)
    cols_to_drop = [c[0] for c in cols_with_nr_of_trues if c[1] < min_freq]  # i c!=change
    df.drop(columns=cols_to_drop, axis=1, inplace=True)
    return df


def drop_instances_without_features(df):
    df = df[(df.drop(columns=["Market_change", "Tweet_sentiment", "Text"]).T != 0).any()]
    return df


def count_nr_of_feature_occurencies(features):  # todo test
    return [(col, (features.loc[features[col] == True, col].count())) for col in features]


def remove_features(df, features_to_leave):
    cols_to_leave = features_to_leave + ["Tweet_sentiment", "Market_change", "Text"]
    cols_to_drop = [c for c in list(df) if c not in cols_to_leave]
    feats_not_in_df = [c for c in cols_to_leave if c not in list(df)]
    if feats_not_in_df:
        raise Exception("There are {0} selected features that are not in the dataset: {1}".format(len(feats_not_in_df), feats_not_in_df))
    df.drop(columns=cols_to_drop, axis=1, inplace=True)
    return df


def get_pos_and_neg_tweets_with_sentiment_from_file():
    pos, neg = [], []
    with open(CORPUS_FILE, 'r', encoding='utf8') as f:
        reader = csv.reader(f, delimiter=",")
        try:
            for line in reader:
                content, sentiment = line[1], line[3]
                if sentiment == "pos":
                    pos.append((content, sentiment))
                elif sentiment != "neg":
                    raise Exception("Error while reading sentiment in line: {}", line)  # TODO test it
                else:
                    neg.append((content, sentiment))
        except IndexError:
            pass
        return pos, neg


def get_train_and_test(folds, k_run):
    """
    Join train folds and return train and test datasets
    >>> get_train_and_test([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10]], 1)
    ([5, 6, 7, 8, 9, 10], [1, 2, 3, 4])
    >>> get_train_and_test([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10]], 3)
    ([1, 2, 3, 4, 5, 6, 7, 8], [9, 10])
    """
    train_data = folds[:]
    test_data = train_data.pop(k_run - 1)
    train_data = [item for fold_items in train_data for item in fold_items]
    return train_data, test_data


def split_pos_and_neg_into_folds(pos, neg, n, random=True):
    """
    Returns folded pos and neg datasets
    >>> split_pos_and_neg_into_folds(range(1,11), range(21, 31), 3, False)
    ([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10]], [[21, 22, 23, 24], [25, 26, 27, 28], [29, 30]])
    """
    pos_folds, neg_folds = [], []
    set_size = len(pos)
    fold_size = ceil(set_size / n)
    indices = list(range(0, set_size))
    if random:
        shuffle(indices)
    while indices:
        fold_indices, indices = indices[:fold_size], indices[fold_size:]
        pos_folds.append([pos[x] for x in fold_indices])
        neg_folds.append([neg[x] for x in fold_indices])
    return pos_folds, neg_folds


def get_train_and_test_data_for_k_run(pos_folds, neg_folds, k_run):
    """
    >>> p = [[(1,1)], [(2,2)], [(3,3)]]
    >>> n = [[(4,4)], [(4,4)], [(4,4)]]
    >>> get_train_and_test_data_for_k_run(p, n, 1)
    ([(2, 2), (3, 3), (4, 4), (4, 4)], [(1, 1), (4, 4)])
    """
    pos_train, pos_test = get_train_and_test(pos_folds, k_run)
    neg_train, neg_test = get_train_and_test(neg_folds, k_run)
    return pos_train + neg_train, pos_test + neg_test


if __name__ == "__main__":
    import doctest

    doctest.testmod()
