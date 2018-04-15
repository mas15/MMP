import pandas as pd
from functools import total_ordering

pd.set_option('display.width', 1500)
pd.options.display.max_colwidth = 1000


@total_ordering
class ModelTrainingResult:
    def __init__(self, test_accuracy=0, train_accuracy=0, df=None):
        self.test_accuracy = test_accuracy
        self.train_accuracy = train_accuracy
        self.df = df
        if df is not None:
            all_columns = list(self.df.columns)
            all_columns.remove("Market_change")
            self.features = all_columns
            self.nr_tweets = self.df["Market_change"].size
            self.base_rate_accuracy = zero_r(self.df)
        else:
            self.features, self.nr_tweets, self.base_rate_accuracy = [], 0, 0

    @property
    def nr_features(self):
        return len(self.features)

    @property
    def base_rate_accuracy_diff(self):
        return self.test_accuracy - self.base_rate_accuracy

    def __lt__(self, other):  # todo zamienic i zobaczyc accuracies
        return (self.base_rate_accuracy_diff, self.nr_features, self.test_accuracy) < \
               (other.base_rate_accuracy_diff, other.nr_features, other.test_accuracy)

    def __eq__(self, other):
        return (self.base_rate_accuracy_diff, self.nr_features, self.test_accuracy) == \
               (other.base_rate_accuracy_diff, other.nr_features, other.test_accuracy)


def zero_r(df): # todo zaokragalac # TODO co jak size =0 ?
    """
    >>> df = pd.DataFrame({"Text": [1, 2, 3, 4, 5], "Market_change":["Up", "Up", "Down", "NC", "Up"]})
    >>> zero_r(df)
    0.6
    """
    return df["Market_change"].value_counts().max() / df["Market_change"].size
