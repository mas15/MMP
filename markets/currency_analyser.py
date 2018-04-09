from markets.predicting_model_builder import ModelTrainer
from markets.association import build_df_with_tweets_and_effect, save_sifted_tweets_with_date
from markets.rules import extract_rules_to_file
from markets.main_model import MarketPredictingModel

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

    def load(self):
        self._predicting_model = MarketPredictingModel(self._currency)
        filename = get_predicting_model_filename(self._currency)  # todo sprawdzic czy jest plik i logować
        self._predicting_model.load(filename)

    def analyse(self):
        # todo test co jak nie ma pliku?

        prices_filename = get_currency_prices_filename(self._currency)
        graph_filename = get_graph_filename(self._currency)
        rules_filename = get_rules_filename(self._currency)
        selected_features_filename = get_selected_features_for_currency_filename(self._currency)
        model_filename = get_predicting_model_filename(self._currency)

        tweets_with_affect_df = build_df_with_tweets_and_effect(ALL_TWEETS_FILE, prices_filename)
        training_result = build_main_model_to_predict_markets(tweets_with_affect_df, model_filename, selected_features_filename)
        self._predicting_model = training_result.model

        save_sifted_tweets_with_date(training_result.df, ALL_TWEETS_FILE, prices_filename, graph_filename)

        print("Model build for {0}".format(self._currency))

        features_df = training_result.df.drop(columns=["Text", "Tweet_sentiment", "Market_change"], axis=1)
        extract_rules_to_file(features_df, rules_filename)

        return training_result

    def get_rules_data(self):
        rules_data = pd.read_csv(get_rules_filename(self._currency))
        return rules_data.to_dict("split")

    def get_most_coefficient_features(self):
        # todo sprawdzic czy jest model
        return self._predicting_model.get_most_coefficient_features()

    def analyse_tweet(self, text):
        # todo sprawdzic czy jest model
        return self._predicting_model.analyse(text)

    def get_graph_data(self):  # czy to dobrze tutaj?
        graph_data = pd.read_csv(get_graph_filename(self._currency))
        tweets_per_date = dict(zip(graph_data.Date, graph_data.Text))
        dates = graph_data["Date"].values.tolist()
        prices = graph_data["Open"].values.tolist()

        return dates, prices, tweets_per_date


def build_main_model_to_predict_markets(df, model_save_filename, selected_features_filename=None): # todo nie potrzebna funkcja
    trainer = ModelTrainer()
    training_result = trainer.train(df, selected_features_filename)
    training_result.model.save(model_save_filename)
    return training_result


def get_selected_features_for_currency_filename(currency):
    return os.path.join(DATA_PATH, currency + "_ready_selected.txt")


def get_rules_filename(currency):
    return os.path.join(DATA_PATH, currency + "_rules.csv")


def get_graph_filename(currency):
    return os.path.join(DATA_PATH, currency + "_graph_data.csv")


def get_currency_prices_filename(currency):
    return os.path.join(DATA_PATH, currency + "Index.csv")


def get_predicting_model_filename(currency):
    return os.path.join(PICKLED_MODEL_PATH, PREDICTING_MODEL_PREFIX + currency + ".pickle")


if __name__ == '__main__':
    for c in ["EUR", "MEX", "USD"]:
        analyser = CurrencyAnalyser(c)
        analyser.analyse()
