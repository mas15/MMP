from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
import os
import pandas as pd
from markets.association import TWEETS_WITH_MARKET_CHANGE
from markets.sentiment import SentimentAnalyser
from markets.feature_extractor import FeatureExtractor
from markets.feature_selection import get_frequent_features, get_best_features_from_file
from markets.helpers import get_x_y_from_df, remove_features, move_column_to_the_end, mark_features, \
    drop_instances_without_features, drop_infrequent_features

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


class MarketPredictingModel:
    def __init__(self):
        self.model = MultinomialNB()  # LogisticRegressionCV(random_state=123, cv=10, Cs=3)
        self.extr = FeatureExtractor(min_keyword_frequency=4)
        self.features = []
        self.sent = SentimentAnalyser()  # todo ogarnac to
        self.sent.load()

    def trainnn(self, df):
        results = []

        self.extr.build_vocabulary(df["Text"].tolist())
        df = self.extract_features(df)
        best_accuracy, best_k, best_features = (0, 0), 0, []

        features = get_frequent_features(df)
        df = self.filter_features(df, features)

        # for features, k_features in get_k_best_features(df, 30, 300):
        for features, k_features in get_best_features_from_file("data/attr_po_6_wr_nb_bf_nc2"):
            print(k_features)
            sifted_df = self.filter_features(df.copy(), features)  # remove not needed, mark other etc
            accuracy = self._train_classifier(sifted_df)
            print("Trained on {0} features and {1} objects, got {2} accuracy".format(k_features, sifted_df.shape[0],
                                                                                     accuracy))
            if accuracy > best_accuracy:
                best_k, best_features = k_features, features

            zero_r_accu_diff = zero_r(sifted_df) - accuracy[0]
            results.append((k_features, zero_r_accu_diff, accuracy[0]))

        print("Best accuracy for {0} features: {1}".format(best_k, best_features))
        sifted_df = self.filter_features(df, best_features)
        self._train_classifier(sifted_df)
        print(results)

    def _train_classifier(self, df):
        sum_train, sum_test = 0, 0
        for n_run in range(1, 31):
            accu_on_test, accu_on_train = self.train(df, n_run)

            sum_test += accu_on_test
            sum_train += accu_on_train

        return sum_test / 30, sum_train / 30

    def extract_features(self, df):  # get features vector?
        df = mark_features(self.extr, df)
        df = calculate_sentiment(df, self.sent)
        df = move_column_to_the_end(df, "Market_change")
        return df

    def filter_features(self, df, features):  # todo one tu nie pasuja?
        sifted_df = remove_features(df, features)  # todo rename remove features?
        self.extr.set_features(features)
        sifted_df = mark_features(self.extr, sifted_df)
        sifted_df = drop_instances_without_features(sifted_df)
        return sifted_df

    def train(self, df, random_state):
        x, y = get_x_y_from_df(df)
        sum_test_accuracy, sum_train_accuracy, model = 0, 0, None

        kf = StratifiedKFold(n_splits=10, random_state=random_state, shuffle=True)
        for train_index, test_index in kf.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            self.model.fit(x_train, y_train)

            accu_on_train, misclass_on_train = self.test_model_on_dataset(x_train, y_train)  # nie slac self
            accu_on_test, misclass_on_test = self.test_model_on_dataset(x_test, y_test)

            # indexes_of_mis_train = get_indexes_before_splitting(train_index, misclass_on_train)
            # indexes_of_mis_test = get_indexes_before_splitting(test_index, misclass_on_test)

            # for i in indexes_of_mis_train.tolist() + indexes_of_mis_test.tolist():
            #     misclassified_objects[i] += 1

            sum_test_accuracy += accu_on_test
            sum_train_accuracy += accu_on_train

        return sum_test_accuracy / 10, sum_train_accuracy / 10  # todo najlepszy z 10

    def test_model_on_dataset(self, x, y):
        predicted = self.model.predict(x)
        accuracy = accuracy_score(y, predicted)
        misclassified_objects = get_misclassified_on_set(y, predicted)
        return accuracy, misclassified_objects

    def analyse(self, text):  # todo co jak nie ma modelu
        df = pd.DataFrame({'Text': [text]})
        df = self.extract_features(df)
        df.drop(columns=["Text"], inplace=True)
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
            pickle.dump(self.extr, f)
            pickle.dump(self.features, f)

    def load(self):
        with open(MARKET_PREDICTING_MODEL_FILE, "rb") as f:
            self.model = pickle.load(f)
            self.extr = pickle.load(f)
            self.features = pickle.load(f)

    def get_most_coefficient_features(self):
        result = dict()
        for i, target in enumerate(self.model.classes_):  # todo check if nr feats = coef
            feats = sorted(zip(self.features, self.model.coef_[i]), key=lambda t: t[1])
            result[target] = feats
        return result


def calculate_sentiment(tweets_df, sent):
    tweets_df["Tweet_sentiment"] = tweets_df["Text"].apply(sent.predict_score)
    return tweets_df


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
    features.drop(columns=["Tweet_sentiment"], inplace=True) # todo tutaj text?
    found_features = drop_infrequent_features(features, 1) # todo pozbyc sie tej funkcji
    result["features"] = found_features.columns.tolist()
    result["sentiment"] = "Positive" if sentiment_value > 0.5 else "Negative"
    return result


if __name__ == '__main__':
    df = pd.read_csv(TWEETS_WITH_MARKET_CHANGE)
    model = MarketPredictingModel()
    model.trainnn(df)
    # model.save()
    # model.load()
    # print(model.get_most_coefficient_features())
    # print(model.analyse("Bad bad Mexicans")) # todo nie przewiduje po zmienionym tsh
