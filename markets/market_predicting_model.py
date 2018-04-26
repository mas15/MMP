import numpy as np
import pickle
from collections import Counter
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from markets import utils
from collections import namedtuple


ModelTrainingResult = namedtuple('ModelTrainingResult', 'test_accuracy train_accuracy base_rate_accuracy')

N_MOST_COEFFICIENT_FEATURES = 20


def get_zero_r_from_y(y):
    """
    >>> get_zero_r_from_y(np.array(["Up", "Up", "Down", "NC", "Up"]))
    0.6
    """
    counted = Counter(y)
    return counted.most_common()[0][1] / y.size


class AnalysisResult:
    def __init__(self, probabilities, sentiment_value, features):
        self.probabilities = probabilities
        self.sentiment_value = sentiment_value
        self.features = features
        self.prediction = max(self.probabilities, key=self.probabilities.get)

    def combine_with(self, other):
        self.probabilities = {t: (v + other.probabilities[t]) * 0.5 for t, v in self.probabilities.items()}
        self.probabilities = {t: round(v, 2) for t, v in self.probabilities.items()}  # round after float operations
        self.sentiment_value = (self.sentiment_value + other.sentiment_value) * 0.5
        self.prediction = max(self.probabilities, key=self.probabilities.get)

    def to_dict(self):
        result = dict(self.probabilities)
        result["Sentiment"] = "Positive" if self.sentiment_value > 0.5 else "Negative"
        result["Features"] = ", ".join(self.features) if self.features else "No features found in the tweet"
        result["Prediction"] = self.prediction if self.prediction != "NC" else "No change"
        return result


def format_result(probabilities, dataset):
    sentiment_value = dataset.get_sentiment()[0]
    features = dataset.get_marked_features()
    return AnalysisResult(probabilities, sentiment_value, features)


class MarketPredictingModel:
    def __init__(self, main_model=None, rest_model=None):
        self.main_model = main_model or Classifier()
        self.rest_model = rest_model or Classifier()
        self.all_features = []
        self.main_features = []

    def train(self, main_df, all_df, random_state=1, k_folds=10):
        self.all_features = all_df.features
        self.main_features = main_df.features

        x, y = main_df.get_x_y()
        main_result = self.main_model.train(x, y, random_state, k_folds)

        x, y = all_df.get_x_y()
        all_result = self.rest_model.train(x, y, random_state, k_folds)
        return main_result, all_result

    def analyse(self, tweet_dataset, sifted_tweet_dataset):
        if sifted_tweet_dataset.get_marked_features():
            return self._analyse_on_model(sifted_tweet_dataset, self.main_model)
        elif tweet_dataset.get_marked_features():
            return self._analyse_on_model(tweet_dataset, self.rest_model)

        main_result = self._analyse_on_model(sifted_tweet_dataset, self.main_model)
        rest_result = self._analyse_on_model(tweet_dataset, self.rest_model)
        main_result.combine_with(rest_result)
        return main_result

    @staticmethod
    def _analyse_on_model(dataset, model):
        x = dataset.get_x()
        probabilities = model.analyse(x)
        return AnalysisResult(probabilities, dataset.get_sentiment()[0], dataset.get_marked_features())

    def save(self, model_filename):
        with open(model_filename, "wb") as f:
            pickle.dump(self.main_model, f)
            pickle.dump(self.rest_model, f)
            pickle.dump(self.all_features, f)
            pickle.dump(self.main_features, f)

    def load(self, model_filename):
        with open(model_filename, "rb") as f:
            self.main_model = pickle.load(f)
            self.rest_model = pickle.load(f)
            self.all_features = pickle.load(f)
            self.main_features = pickle.load(f)

    def get_most_coefficient_features(self):
        result = dict()
        for (features_with_scores, target) in self.main_model.get_coefficient_features():
            features_names = self.main_features+["Tweet_sentiment"]
            features_names_with_scores = zip(features_names, features_with_scores)
            sorted_features = sorted(features_names_with_scores, key=lambda t: -t[1])
            sorted_features = sorted_features[:N_MOST_COEFFICIENT_FEATURES]
            sorted_features = [(name, -N_MOST_COEFFICIENT_FEATURES/value) for name, value in sorted_features]
            sorted_features = [(name, round(value, 2)) for name, value in sorted_features]
            result[target] = sorted_features
        return result


class Classifier:
    def __init__(self, model=None):
        self.model = model or MultinomialNB()
        #self.model = model or LogisticRegressionCV(random_state=123, cv=10, Cs=3)

    def get_coefficient_features(self):
        return zip(self.model.coef_, self.model.classes_)

    def train(self, x, y, nr_of_runs=30, k_folds=10):
        sum_train, sum_test = 0, 0
        for n_run in range(nr_of_runs):
            test_accuracy, train_accuracy = self._train(x, y, n_run + 1, k_folds)

            sum_test += test_accuracy
            sum_train += train_accuracy

        result = ModelTrainingResult(sum_test / nr_of_runs, sum_train / nr_of_runs, get_zero_r_from_y(y))
        print("Accuracy {0} ({1}) and zeroR {2}".format(result.test_accuracy, result.train_accuracy, result.base_rate_accuracy))
        return result

    def _train(self, x, y, random_state, k_folds):
        sum_test_accuracy, sum_train_accuracy, = 0, 0

        for x_train, x_test, y_train, y_test in utils.k_split(x, y, k_folds, random_state):
            self.model.fit(x_train, y_train)

            accu_on_test = self.test_model_on_dataset(x_test, y_test)
            accu_on_train = self.test_model_on_dataset(x_train, y_train)

            sum_test_accuracy += accu_on_test
            sum_train_accuracy += accu_on_train

        return sum_test_accuracy / k_folds, sum_train_accuracy / k_folds

    def test_model_on_dataset(self, x, y):
        predicted = self.model.predict(x)
        accuracy = metrics.accuracy_score(y, predicted)
        return accuracy

    def analyse(self, x):  # todo co jak nie ma modelu
        propabs_vals = self.model.predict_proba(x)
        propabs_vals = propabs_vals[0].tolist()
        propabs = dict(zip(self.model.classes_, propabs_vals))
        return propabs

