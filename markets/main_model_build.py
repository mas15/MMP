import numpy as np
import pickle
import pandas as pd
import os
from collections import Counter
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from markets.sentiment import SentimentAnalyser, calculate_sentiment
from markets.feature_extractor import FeatureExtractor
from markets.helpers import get_x_y_from_df, remove_features, move_column_to_the_end, mark_features, \
    drop_instances_without_features
from markets import helpers

PICKLED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "pickled_models")
PREDICTING_MODEL_PREFIX = "assoc_model"


def get_predicting_model_filename(currency):
    return os.path.join(PICKLED_MODEL_PATH, PREDICTING_MODEL_PREFIX + currency + ".pickle")


class AssociationDataProcessor:
    def __init__(self, features=None, extr=None, sent=None):
        self.extr = extr or FeatureExtractor(min_keyword_frequency=4)
        if features:
            self.extr.set_features(features)
        self.sent = sent or SentimentAnalyser()
        self.sent.load()

    def extract_features(self, df):  # get features vector?
        if not self.extr.features:  # todo has_features
            self.extr.build_vocabulary(df["Text"].tolist())

        df = mark_features(self.extr, df)
        df = calculate_sentiment(df, self.sent)

        if "Market_change" in list(df):
            df = move_column_to_the_end(df, "Market_change")
        return df

    def filter_features(self, df, features):
        features_to_leave = features + ["Tweet_sentiment", "Market_change", "Text"]
        sifted_df = remove_features(df, features_to_leave)

        self.extr.set_features(features)
        sifted_df = mark_features(self.extr, sifted_df)
        sifted_df = drop_instances_without_features(sifted_df)
        return sifted_df

    def process_text(self, text):
        df = pd.DataFrame({'Text': [text]})
        df = self.extract_features(df)
        df.drop(columns=["Text"], inplace=True)
        return df


class ProvisionalPredictingModel:
    def __init__(self, model=None):
        self.model = model or MultinomialNB()  # LogisticRegressionCV(random_state=123, cv=10, Cs=3)

    def train(self, df, random_state=1, k_folds=10):
        sum_test_accuracy, sum_train_accuracy, = 0, 0

        x, y = get_x_y_from_df(df)
        for x_train, x_test, y_train, y_test in helpers.k_split(x, y, k_folds, random_state):
            self.model.fit(x_train, y_train)

            accu_on_test, misclass_on_test = self.test_model_on_dataset(x_test, y_test)
            accu_on_train, misclass_on_train = self.test_model_on_dataset(x_train, y_train)  # nie slac self

            # indexes_of_mis_train = get_indexes_before_splitting(train_index, misclass_on_train)
            # indexes_of_mis_test = get_indexes_before_splitting(test_index, misclass_on_test)
            # misclass_in_all_cvs.update(indexes_of_mis_train.tolist())
            # misclass_in_all_cvs.update(indexes_of_mis_test.tolist())

            sum_test_accuracy += accu_on_test
            sum_train_accuracy += accu_on_train

        return sum_test_accuracy / k_folds, sum_train_accuracy / k_folds  # todo najlepszy z 10

    def test_model_on_dataset(self, x, y):
        predicted = self.model.predict(x)
        accuracy = metrics.accuracy_score(y, predicted)
        misclassified_objects = get_misclassified_on_set(y, predicted)
        return accuracy, misclassified_objects


class MarketPredictingModel(ProvisionalPredictingModel):
    def __init__(self, currency, features=None, model=None):
        super(MarketPredictingModel, self).__init__(model)
        self._currency = currency
        self.features = features or []

    def analyse(self, text):  # todo co jak nie ma modelu
        df = AssociationDataProcessor(self.features).process_text(text)
        prediction, propabs = self._predict(df)
        result = put_results_in_dict(prediction, propabs, df)
        return result

    def _predict(self, features):
        result = self.model.predict(features)
        result = str(result[0])
        propabs_vals = self.model.predict_proba(features)
        propabs_vals = propabs_vals[0].tolist()
        propabs = dict(zip(self.model.classes_, propabs_vals))
        return result, propabs

    def save(self):
        with open(get_predicting_model_filename(self._currency), "wb") as f:
            pickle.dump(self.model, f)
            pickle.dump(self.features, f)

    def load(self):
        with open(get_predicting_model_filename(self._currency), "rb") as f:
            self.model = pickle.load(f)
            self.features = pickle.load(f)

    def get_most_coefficient_features(self):
        if len(self.model.coef_) != len(self.model.classes_):
            raise Exception("Different number of model features than coefs.")

        result = dict()
        for i, target in enumerate(self.model.classes_):
            feats = sorted(zip(self.features, self.model.coef_[i]), key=lambda t: t[1])
            result[target] = feats
        return result


def zero_r(df):
    """
    >>> df = pd.DataFrame({"Text": [1, 2, 3, 4, 5], "Market_change":["Up", "Up", "Down", "NC", "Up"]})
    >>> zero_r(df)
    0.6
    """
    return df["Market_change"].value_counts().max() / df["Market_change"].size


def print_misclassified(df, misclassified_objects):
    print()
    print("misclassified_objects")
    for object_index, number_of_misclassifiations in sort_misclassified(misclassified_objects):
        print(number_of_misclassifiations)
        row = df.iloc[object_index][["Text", "Market_change"]]
        print(row["Market_change"] + " " + row["Text"])
    print()


def sort_misclassified(misclassified_objects):
    misclassified_objects = sorted(misclassified_objects.items(), key=lambda x: x[1], reverse=True)
    misclassified_objects = [m for m in misclassified_objects if m[1]]
    return misclassified_objects


def get_indexes_before_splitting(before, after):
    return before[after]


def get_misclassified_on_set(y, predicted):
    misclassified_objects = np.where(y != predicted)
    return misclassified_objects


def put_results_in_dict(prediction, propabs, features):
    result = dict(propabs)
    result["prediction"] = prediction
    sentiment_value = features["Tweet_sentiment"].iloc[0]
    features.drop(columns=["Tweet_sentiment"], inplace=True)  # todo tutaj text?
    result["features"] = features.columns[features.any()].tolist()  # todo test czy dziala po zmianie
    result["sentiment"] = "Positive" if sentiment_value > 0.5 else "Negative"
    return result
