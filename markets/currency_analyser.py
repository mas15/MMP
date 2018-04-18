from markets.feature_selection import select_features, filter_features
from markets.association import build_df_with_tweets_and_affect, save_sifted_tweets_with_date
from markets.rules import extract_rules_to_file, read_rules_sets
from markets.dataset import TweetsDataSet
from markets.market_predicting_model import DoubleMarketPredictingModel
from markets.tweets_features_extraction import TWEETS_WITH_FEATURES_FILENAME, build_dataset_with_one_tweet

import pandas as pd
import os
from collections import namedtuple

PICKLED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "pickled_models")
PREDICTING_MODEL_PREFIX = "assoc_model"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
ALL_TWEETS_FILE = os.path.join(DATA_PATH, "all_tweets.csv")

AnalyseResult = namedtuple('AnalyseResult',
                           'currency test_accuracy train_accuracy base_rate_accuracy nr_features nr_tweets')


class CurrencyAnalyser:
    def __init__(self, currency):
        self._currency = currency
        self._predicting_model = None  # todo czy to dobrze?
        self.selected_features_filename = os.path.join(DATA_PATH, self._currency + "_ready_selected.txt")
        self.rules_filename = os.path.join(DATA_PATH, self._currency + "_rules.csv")
        self.graph_filename = os.path.join(DATA_PATH, self._currency + "_graph_data.csv")
        self.currency_prices_filename = os.path.join(DATA_PATH, self._currency + "Index.csv")
        self.model_filename = os.path.join(PICKLED_MODEL_PATH, PREDICTING_MODEL_PREFIX + self._currency + ".pickle")

    def load(self):
        self._predicting_model = DoubleMarketPredictingModel()
        # todo sprawdzic czy jest plik i logować
        self._predicting_model.load(self.model_filename)

    def analyse(self):
        # todo test co jak nie ma pliku?

        tweets_with_affect_df = build_df_with_tweets_and_affect(TWEETS_WITH_FEATURES_FILENAME, self.currency_prices_filename)
        sifted_tweets_df = select_features(tweets_with_affect_df, self.selected_features_filename)
        training_result = self.build_main_model_to_predict_markets(sifted_tweets_df, tweets_with_affect_df)
        save_sifted_tweets_with_date(sifted_tweets_df, ALL_TWEETS_FILE, self.currency_prices_filename, self.graph_filename)

        print("Model build for {0}".format(self._currency))

        features_df = sifted_tweets_df.get_features_df()
        extract_rules_to_file(features_df, self.rules_filename)

        return training_result

    def get_rules_data(self):
        return read_rules_sets(self.rules_filename)

    def get_most_coefficient_features(self):
        # todo sprawdzic czy jest model
        result = self._predicting_model.get_most_coefficient_features()
        return result

    def analyse_tweet(self, text):
        if self._predicting_model is None:
            raise Exception("Model has not been built yet")
        tweet_dataset = build_dataset_with_one_tweet(text, self._predicting_model.all_features)
        sifted_tweet_dataset = filter_features(tweet_dataset, self._predicting_model.main_features, False)
        result = self._predicting_model.analyse(tweet_dataset, sifted_tweet_dataset)
        return result.to_dict()

    def get_graph_data(self):  # czy to dobrze tutaj?
        graph_data = pd.read_csv(self.graph_filename)
        tweets_per_date = dict(zip(graph_data.Date, graph_data.Text))
        dates = graph_data["Date"].values.tolist()
        prices = graph_data["Open"].values.tolist()

        return dates, prices, tweets_per_date

    def build_main_model_to_predict_markets(self, main_df, all_df):
        self._predicting_model = DoubleMarketPredictingModel()
        main_result, rest_result = self._predicting_model.train(main_df, all_df)  # todo use rest_result
        self._predicting_model.save(self.model_filename)

        return AnalyseResult(self._currency,
                             main_result.test_accuracy,
                             main_result.train_accuracy,
                             main_result.base_rate_accuracy,
                             len(main_df.features),
                             len(main_df.get_all_tweets()))


if __name__ == '__main__':
    for c in ["USD", "EUR", "MEX"]:
        analyser = CurrencyAnalyser(c)
        analyser.analyse()
        print(analyser.analyse_tweet("Fuckin insurance companies"))
        print(analyser.analyse_tweet("lalal abc"))
        print(analyser.analyse_tweet("siema co tam"))
