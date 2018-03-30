from sklearn.naive_bayes import MultinomialNB
import os
import pandas as pd
from sklearn.feature_selection import RFECV
from markets.helpers import get_x_y_from_df, count_nr_of_feature_occurencies

ASSOCIATION_MODEL_FILE = os.path.join(os.path.dirname(__file__), "assoc_model.pickle")
FEATURES_WITH_EFFECT_FILE = os.path.join(os.path.dirname(__file__), "data/features_with_effect.csv")
pd.set_option('display.width', 1500)
MIN_FEATURE_OCCURENCIES = 6


class FeatureSelector:
    def __init__(self, df):
        self.df = df
        self.x, self.y = get_x_y_from_df(df)
        self.features_names = df.drop(columns=['Market_change', 'Text']).columns.tolist()
        self.sorted_features = []
        pass

    def select_k_best_features(self, k_features):
        if not self.sorted_features:
            self.sorted_features = self._sort_features_by_rank()

        selected = self.sorted_features[:k_features]
        return selected

    def _sort_features_by_rank(self):
        model = MultinomialNB()
        rfe = RFECV(model, 1, cv=10, verbose=1, n_jobs=-1)
        rfe = rfe.fit(self.x, self.y)
        ranking = [int(score) for score in rfe.ranking_.tolist()]

        sorted_indexes = get_indexes_sorted_by_score(ranking)
        sorted_features = [self.features_names[i] for i in sorted_indexes]
        return sorted_features


def get_indexes_sorted_by_score(scores):
    sorted_by_score = sorted(zip(range(len(scores)), scores), key=lambda t: t[1])
    return [i for i, score in sorted_by_score]


def get_frequent_features(df, min_freq=MIN_FEATURE_OCCURENCIES):
    features = df.drop(columns=["Market_change", "Tweet_sentiment", "Text"], errors="ignore")
    cols_with_nr_of_trues = count_nr_of_feature_occurencies(features)
    frequent_features = [c[0] for c in cols_with_nr_of_trues if c[1] > min_freq]  # i c!=change
    return frequent_features


def get_k_best_features(df, k_min, k_max):
    selector = FeatureSelector(df)
    for k in range(k_min, k_max):
        yield selector.select_k_best_features(k), k


def get_best_features_from_file(filename):
    features = [line.strip() for line in open(filename, 'r')]
    yield features, len(features)