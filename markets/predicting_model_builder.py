import os
import pandas as pd
from markets.main_model import AssociationDataProcessor, MarketPredictingModel
from markets.feature_selection import get_frequent_features, get_best_features_from_file, get_k_best_features
from functools import total_ordering

pd.set_option('display.width', 1500)
pd.options.display.max_colwidth = 1000


@total_ordering
class ModelTrainingResult:
    def __init__(self, model=None, test_accuracy=0, train_accuracy=0, df=None):
        self.model = model
        self.test_accuracy = test_accuracy
        self.train_accuracy = train_accuracy
        self.df = df
        if df is not None:
            all_columns = list(self.df.columns)
            all_columns.remove("Market_change")
            self.features = all_columns
            self.nr_tweets = self.df["Market_change"].size
            self.zero_r = self.df["Market_change"].value_counts().max() / self.df["Market_change"].size
        else:
            self.features, self.nr_tweets, self.zero_r = [], 0,  0

    @property
    def nr_features(self):
        return len(self.features)

    def __lt__(self, other):  # todo zamienic i zobaczyc accuracies
        return (self.test_accuracy, self.test_accuracy - self.zero_r, self.nr_features) < \
               (other.test_accuracy, other.test_accuracy - other.zero_r, other.nr_features)

    def __eq__(self, other):
        return (self.test_accuracy, self.test_accuracy - self.zero_r, self.nr_features) == \
               (other.test_accuracy, other.test_accuracy - other.zero_r, other.nr_features)


class ModelTrainer:
    def __init__(self, df_processor=None):
        self.df_processor = df_processor or AssociationDataProcessor()

    def train(self, df, features_filename=None):
        best_result = ModelTrainingResult()

        df = self.df_processor.extract_features(df)
        features = get_frequent_features(df)
        df = self.df_processor.filter_features(df, features)

        for features in get_features_iterator(df, features_filename):
            sifted_df = self.df_processor.filter_features(df.copy(), features)

            training_result = self._train(sifted_df, features)

            if training_result > best_result:
                best_result = training_result

        print("Best accuracy ({0} for {1} features: {2}".format(best_result.test_accuracy, best_result.nr_features, best_result.features))
        return best_result

    def _train(self, df, features):
        result = ModelTrainingResult(df=df)
        result.test_accuracy, result.train_accuracy, result.model = self._train_with_different_seeds(df, features)
        return result

    @staticmethod
    def _train_with_different_seeds(df, features): # todo decorator
        sum_train, sum_test, model = 0, 0, None
        for n_run in range(1, 31):
            model = MarketPredictingModel(features)
            test_accuracy, train_accuracy = model.train(df, n_run)

            sum_test += test_accuracy
            sum_train += train_accuracy

        return sum_test / 30, sum_train / 30, model


def zero_r(df):
    """
    >>> df = pd.DataFrame({"Text": [1, 2, 3, 4, 5], "Market_change":["Up", "Up", "Down", "NC", "Up"]})
    >>> zero_r(df)
    0.6
    """
    return df["Market_change"].value_counts().max() / df["Market_change"].size


def get_features_iterator(df, selected_features_filename=None):  # todo test
    if selected_features_filename:
        if os.path.isfile(selected_features_filename):
            return get_best_features_from_file(selected_features_filename)
        # lgo here no file
    return get_k_best_features(df, 110, 115) # było 100-130 a kiedyś i więcej

