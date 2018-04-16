import numpy as np
import pickle
from collections import Counter
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from markets.helpers import get_x_y_from_df
from markets import helpers
from collections import namedtuple


ModelTrainingResult = namedtuple('ModelTrainingResult', 'test_accuracy train_accuracy base_rate_accuracy')


def get_zero_r_from_y(y):
    """
    >>> get_zero_r_from_y(np.array(["Up", "Up", "Down", "NC", "Up"]))
    0.6
    """
    counted = Counter(y)
    return counted.most_common()[0][1] / y.size


class AnalysisResult:
    def __init__(self, prediction, sentiment, features, propabs):
        self.up = propabs["Up"]
        self.down = propabs["Down"]
        self.nc = propabs["NC"]
        self.sentiment = sentiment
        self.features = features
        self.prediction = prediction

    def combine_with(self, other):
        # self.prediction = z propabsow
        pass


class DoubleMarketPredictingModel:
    def __init__(self, features=None, model=None):
        self.features = features or []
        self.main_model = MarketPredictingModel()
        self.rest_model = MarketPredictingModel()

    def train(self, main_df, all_df, random_state=1, k_folds=10):
        main_result = self.main_model.train(main_df, random_state, k_folds)
        all_result = self.rest_model.train(all_df, random_state, k_folds) # todo z inymi featurami
        return main_result, all_result

    def analyse(self, tweet):
        main_result = self.main_model.analyse(tweet)
        rest_result = self.main_model.analyse(tweet)
        if main_result.features:
            return main_result.result
        return rest_result.combine_with(main_result)

    def save(self, model_filename):
        with open(model_filename, "wb") as f:
            pickle.dump(self.main_model, f)
            pickle.dump(self.rest_model, f)

    def load(self, model_filename):
        with open(model_filename, "rb") as f:
            self.main_model = pickle.load(f)
            self.rest_model = pickle.load(f)

    def get_most_coefficient_features(self):
        return self.main_model.get_most_coefficient_features()


class MarketPredictingModel:
    def __init__(self, features=None, model=None):
        self.features = features or []
        self.model = model or MultinomialNB()  # LogisticRegressionCV(random_state=123, cv=10, Cs=3)

    def train(self, df, nr_of_runs=30, k_folds=10):
        sum_train, sum_test = 0, 0

        x, y = df.get_x_y()
        for n_run in range(nr_of_runs):
            test_accuracy, train_accuracy = self._train(x, y, n_run + 1, k_folds)

            sum_test += test_accuracy
            sum_train += train_accuracy

        result = ModelTrainingResult(sum_test / nr_of_runs, sum_train / nr_of_runs, get_zero_r_from_y(y))
        print("Accuracy {0} ({1}) and zeroR {2}".format(result.test_accuracy, result.train_accuracy,result.base_rate_accuracy))
        return result

    def _train(self, x, y, random_state, k_folds):
        sum_test_accuracy, sum_train_accuracy, = 0, 0

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

    def analyse(self, text):  # todo co jak nie ma modelu
        pass
        # features_extractor = TweetFeaturesExtractor(self.features)
        # df = features_extractor.process_text(text)
        # prediction, propabs = self._predict(df)
        # result = format_result(prediction, propabs, df)
        # return result

    def _predict(self, features):
        result = self.model.predict(features)
        result = str(result[0])
        propabs_vals = self.model.predict_proba(features)
        propabs_vals = propabs_vals[0].tolist()
        propabs = dict(zip(self.model.classes_, propabs_vals))
        return result, propabs

    def get_most_coefficient_features(self):
        if len(self.model.coef_) != len(self.model.classes_):
            raise Exception("Different number of model features than coefs.")

        result = dict()
        for i, target in enumerate(self.model.classes_):
            feats = sorted(zip(self.features, self.model.coef_[i]), key=lambda t: t[1])
            result[target] = feats
        return result


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


def get_misclassified_on_set(y, predicted): # tu cos moze teraz byc nie tak todo
    misclassified_objects = np.where(y != predicted)
    return misclassified_objects


def format_result(prediction, propabs, features):
    sentiment_value = features["Tweet_sentiment"].iloc[0]
    features.drop(columns=["Tweet_sentiment"], inplace=True)  # todo tutaj text?
    features = features.columns[features.any()].tolist()
    sentiment = "Positive" if sentiment_value > 0.5 else "Negative"
    propabs = dict(propabs)
    return AnalysisResult(prediction, sentiment, features, propabs)
