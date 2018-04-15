from markets.feature_selection import select_features
from markets.association import build_df_with_tweets_and_affect, save_sifted_tweets_with_date
from markets.rules import extract_rules_to_file
from markets.market_predicting_model import MarketPredictingModel
from markets.market_predicting_model import DoubleMarketPredictingModel
from markets.tweets_features_extraction import TWEETS_WITH_FEATURES_FILENAME

import pandas as pd
import os

PICKLED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "pickled_models")
PREDICTING_MODEL_PREFIX = "assoc_model"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
ALL_TWEETS_FILE = os.path.join(DATA_PATH, "all_tweets.csv")


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
        self._predicting_model = MarketPredictingModel()
        # todo sprawdzic czy jest plik i logować
        self._predicting_model.load(self.model_filename)

    def analyse(self):
        # todo test co jak nie ma pliku?

        tweets_with_affect_df = build_df_with_tweets_and_affect(TWEETS_WITH_FEATURES_FILENAME, self.currency_prices_filename)
        sifted_tweets_df = select_features(tweets_with_affect_df, self.selected_features_filename)
        training_result, self._predicting_model = self.build_main_model_to_predict_markets(sifted_tweets_df, tweets_with_affect_df)
        save_sifted_tweets_with_date(sifted_tweets_df, ALL_TWEETS_FILE, self.currency_prices_filename, self.graph_filename)

        print("Model build for {0}".format(self._currency))

        features_df = training_result.df.drop(columns=["Text", "Tweet_sentiment", "Market_change"], axis=1)
        extract_rules_to_file(features_df, self.rules_filename)

        return training_result

    def get_rules_data(self):
        rules_data = pd.read_csv(self.rules_filename)
        return rules_data.to_dict("split")

    def get_most_coefficient_features(self):
        # todo sprawdzic czy jest model
        return self._predicting_model.get_most_coefficient_features()

    def analyse_tweet(self, text):
        # todo sprawdzic czy jest model
        return self._predicting_model.analyse(text)

    def get_graph_data(self):  # czy to dobrze tutaj?
        graph_data = pd.read_csv(self.graph_filename)
        tweets_per_date = dict(zip(graph_data.Date, graph_data.Text))
        dates = graph_data["Date"].values.tolist()
        prices = graph_data["Open"].values.tolist()

        return dates, prices, tweets_per_date

    def build_main_model_to_predict_markets(self, main_df, all_df):
        model = DoubleMarketPredictingModel()
        main_result, rest_result = model.train(main_df, all_df)  # todo use rest_result
        model.save(self.model_filename)
        return main_result, model


if __name__ == '__main__':
    for c in ["USD", "EUR", "MEX"]:
        analyser = CurrencyAnalyser(c)
        analyser.analyse()
