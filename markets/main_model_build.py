import numpy as np
import pickle
import os
import pandas as pd
from collections import Counter
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from markets.association import TWEETS_WITH_MARKET_CHANGE, save_sifted_tweets_with_date
from markets.sentiment import SentimentAnalyser, calculate_sentiment
from markets.feature_extractor import FeatureExtractor
from markets.feature_selection import get_frequent_features, get_best_features_from_file
from markets.helpers import get_x_y_from_df, remove_features, move_column_to_the_end, mark_features, \
    drop_instances_without_features
from markets import helpers

MARKET_PREDICTING_MODEL_FILE = os.path.join(os.path.dirname(__file__), "pickled_models/assoc_model.pickle")
pd.set_option('display.width', 1500)
pd.options.display.max_colwidth = 1000


# def build_model(df):
#     feature_selector = FeatureSelector(df)
#     best_model_accu = 0
#     best_k_features = 0
#     best_model = None
#     best_features = []
#
#     for k_feats in range(120, 130):
#         selected_features = feature_selector.select_k_best_features(k_feats)
#         sifted_df = select_features(df, selected_features)
#
#         # sifted_df.drop(columns=['Text'], inplace=True)  # TODO zachowac jakos
#
#         model, accu_on_test, accu_on_train = train_model_many_runs(sifted_df)
#
#         if accu_on_test > best_model_accu:
#             best_model_accu = accu_on_test
#             best_k_features = k_feats
#             best_model = model
#             best_features = selected_features
#         print()
#         print("Accuracy on train: {0}".format(accu_on_train))
#         print("Accuracy on test:  {0}".format(accu_on_test))
#         # print_misclassified(misclassified_objects)
#
#     print("Best accuracy on test:  {0}".format(best_model_accu))
#     print("Best k_features:  {0}".format(best_k_features))
#     print(best_features)
#     print("Best features:  {0}".format(best_features))
#
#     return best_model

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


class ModelTrainer:
    def __init__(self):
        self.df_processor = AssociationDataProcessor()

    def train(self, df):
        best_features = self._find_best_features(df)

        df_with_features = self.df_processor.extract_features(df)
        sifted_df = self.df_processor.filter_features(df_with_features, best_features)

        model = MarketPredictingModel(best_features)
        model.train(sifted_df)

        save_sifted_tweets_with_date(sifted_df)
        return model

    def _find_best_features(self, df):
        best_accuracy, best_k, best_features = (0, 0), 0, []

        df = self.df_processor.extract_features(df)
        features = get_frequent_features(df)
        df = self.df_processor.filter_features(df, features)

        # for features, k_features in get_k_best_features(df, 30, 300):
        for features, k_features in get_best_features_from_file("data/attr_selected_in_weka"):
            print(k_features)
            sifted_df = self.df_processor.filter_features(df.copy(), features)  # remove not needed, mark other etc
            accuracy = self._train_with_different_seeds(sifted_df)
            print("Trained on {0} features and {1} objects, got {2} accuracy".format(k_features, sifted_df.shape[0],
                                                                                     accuracy))

            if accuracy > best_accuracy:
                best_k, best_features = k_features, features

            # zero_r_accu_diff = zero_r(sifted_df) - accuracy[0]

        print("Best accuracy for {0} features: {1}".format(best_k, best_features))
        return best_features

    @staticmethod
    def _train_with_different_seeds(df):
        sum_train, sum_test = 0, 0

        for n_run in range(1, 31):
            model = ProvisionalPredictingModel()
            accu_on_test, accu_on_train = model.train(df, n_run)

            sum_test += accu_on_test
            sum_train += accu_on_train

        return sum_test / 30, sum_train / 30


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
    def __init__(self, features=None, model=None):
        super(MarketPredictingModel, self).__init__(model)
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
        with open(MARKET_PREDICTING_MODEL_FILE, "wb") as f:
            pickle.dump(self.model, f)
            pickle.dump(self.features, f)

    def load(self):
        with open(MARKET_PREDICTING_MODEL_FILE, "rb") as f:
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




def build_main_model_to_predict_markets():
    df = pd.read_csv(TWEETS_WITH_MARKET_CHANGE)
    model = ModelTrainer().train(df)
    model.save()

    #model = MarketPredictingModel()
    model.load()
    print(model.get_most_coefficient_features())
    print(model.analyse("Bad bad Mexicans"))  # todo nie przewiduje po zmienionym tsh


if __name__ == '__main__':
    build_main_model_to_predict_markets()
