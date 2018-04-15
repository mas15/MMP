import numpy as np
from sklearn.model_selection import StratifiedKFold


def k_split(x, y, nr_folds, random_state):  # todo test
    kf = StratifiedKFold(n_splits=nr_folds, random_state=random_state, shuffle=True)
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        yield x_train, x_test, y_train, y_test


def get_x_y_from_df(df):
    target_column = df.columns[-1]
    y = df[target_column].values.ravel()
    x = df.drop(columns=[target_column, "Text"]).values  # todo text jakos ogarnac
    return x, y


def get_x_y_from_list_of_tuples(dataset):
    x, y = zip(*dataset)
    x = np.array(x)
    y = np.array(y)
    return x, y


def move_column_to_the_end(df, col_name):
    cols = list(df)
    cols.append(cols.pop(cols.index(col_name)))
    df = df.reindex(columns=cols)
    return df


def drop_instances_without_features(df):
    df = df[(df.drop(columns=["Market_change", "Tweet_sentiment", "Text"]).T != 0).any()]
    return df


def count_nr_of_feature_occurrences(features):
    return [(col, (features.loc[features[col] == True, col].count())) for col in features]


def remove_features(df, features_to_remove): # todo usunac?
    df.drop(columns=features_to_remove, axis=1, inplace=True)
    return df


def filter_columns(df, features_to_leave):
    sifted_df = df[features_to_leave]
    sifted_df = move_column_to_the_end(sifted_df, "Market_change")
    return sifted_df
