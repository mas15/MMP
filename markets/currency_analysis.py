import os
import pandas as pd
from collections import namedtuple, defaultdict
from markets.feature_selection import select_features, filter_features
from markets.association import build_df_with_tweets_and_affect, save_sifted_tweets_with_date
from markets.rules import extract_rules_to_file, read_rules_sets
from markets.market_predicting_model import MarketPredictingModel
from markets.tweets_features_extraction import TWEETS_WITH_FEATURES_FILENAME, build_dataset_with_one_tweet


PICKLED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "pickled_models")
PREDICTING_MODEL_PREFIX = "assoc_model"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
ALL_TWEETS_FILE = os.path.join(DATA_PATH, "all_tweets.csv")

AnalyseResult = namedtuple('AnalyseResult',
                           'currency test_accuracy train_accuracy base_rate_accuracy nr_features nr_tweets')


class CurrencyAnalyser:
    def __init__(self, currency):
        self._currency = currency
        self._model = None
        self.selected_features_filename = os.path.join(DATA_PATH, self._currency + "_ready_selected.txt")
        self.rules_filename = os.path.join(DATA_PATH, self._currency + "_rules.csv")
        self.graph_filename = os.path.join(DATA_PATH, self._currency + "_graph_data.csv")
        self.currency_prices_filename = os.path.join(DATA_PATH, self._currency + "Index.csv")
        self.model_filename = os.path.join(PICKLED_MODEL_PATH, PREDICTING_MODEL_PREFIX + self._currency + ".pickle")

    def load(self):  # TODO test?
        self._model = MarketPredictingModel()
        if not os.path.isfile(self.model_filename):
            raise Exception("Cannot find a pickled model file")
        self._model.load(self.model_filename)

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
        self._check_if_model_is_build()  # TODO test it
        result = self._model.get_most_coefficient_features()
        return result

    def _check_if_model_is_build(self):
        if self._model is None:
            raise Exception("Model has not been built yet")

    def analyse_tweet(self, text):
        self._check_if_model_is_build()
        tweet_dataset = build_dataset_with_one_tweet(text, self._model.all_features)
        sifted_tweet_dataset = filter_features(tweet_dataset, self._model.main_features, False)
        result = self._model.analyse(tweet_dataset, sifted_tweet_dataset)
        return result.to_dict()

    def get_graph_data(self):
        graph_data = pd.read_csv(self.graph_filename)
        tweets_per_date = defaultdict(list)
        for date, tweet in zip(graph_data.Date, graph_data.Text):
            tweets_per_date[date].append(tweet)

        dates = graph_data["Date"].values.tolist()
        prices = graph_data["Open"].values.tolist()

        return dates, prices, tweets_per_date

    def build_main_model_to_predict_markets(self, main_df, all_df):
        self._model = MarketPredictingModel()
        main_result, rest_result = self._model.train(main_df, all_df)  # todo use rest_result
        self._model.save(self.model_filename)

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
