import os
import pandas as pd
from markets.helpers import save_features_df, DATA_PATH
from markets.main_model import AssociationDataProcessor, MarketPredictingModel, ProvisionalPredictingModel
from markets.association import save_sifted_tweets_with_date, build_df_with_tweets_and_effect
from markets.feature_selection import get_frequent_features, get_best_features_from_file, get_k_best_features

pd.set_option('display.width', 1500)
pd.options.display.max_colwidth = 1000


class ModelTrainer:
    def __init__(self):
        self.df_processor = AssociationDataProcessor()

    def train(self, df, currency): # todo test
        best_features = self._find_best_features(df, currency)

        df_with_features = self.df_processor.extract_features(df)
        sifted_df = self.df_processor.filter_features(df_with_features, best_features)

        model = MarketPredictingModel(currency, best_features)
        model.train(sifted_df)

        save_features_df(sifted_df, currency)  # todo czy to dobre miejsce?
        save_sifted_tweets_with_date(sifted_df, currency)
        return model

    def _find_best_features(self, df, currency):
        best_accuracy, best_k, best_features = (0, 0), 0, []

        df = self.df_processor.extract_features(df)
        features = get_frequent_features(df)
        df = self.df_processor.filter_features(df, features)

        for features, k_features in get_features_iterator(currency, df):
            # print(k_features)
            sifted_df = self.df_processor.filter_features(df.copy(), features)  # remove not needed, mark other etc
            accuracy = self._train_with_different_seeds(sifted_df)
            # print("Trained on {0} features and {1} objects, got {2} accuracy".format(k_features, sifted_df.shape[0], accuracy))

            if accuracy > best_accuracy:
                best_k, best_features, best_accuracy = k_features, features, accuracy

            # zero_r_accu_diff = zero_r(sifted_df) - accuracy[0]

        print("Best accuracy ({0} for {1} features: {2}".format(best_accuracy, best_k, best_features))
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


def get_features_iterator(currency, df):  # todo test
    selected_features_filename = get_selected_features_for_currency_filename(currency)
    if os.path.isfile(selected_features_filename):
        return get_best_features_from_file(selected_features_filename)
    return get_k_best_features(df, 100, 130)


def get_selected_features_for_currency_filename(currency):  # todo test
    return os.path.join(DATA_PATH, currency + "_ready_selected.txt")


def build_main_model_to_predict_markets(tweets_with_markets_filename, currency):
    df = pd.read_csv(tweets_with_markets_filename)
    model = ModelTrainer().train(df, currency)
    model.save()

    # model = MarketPredictingModel(currency)
    model.load()
    print(model.get_most_coefficient_features())
    print(model.analyse("Bad bad Mexicans"))  # todo nie przewiduje po zmienionym tsh


def build_all():
    for currency in ["EUR", "MEX", "USD"]:
        tweets_with_markets_filename = build_df_with_tweets_and_effect(currency)
        build_main_model_to_predict_markets(tweets_with_markets_filename, currency)
        print("Model build for {0}".format(currency))


if __name__ == '__main__':
    build_all()
#    build_main_model_to_predict_markets()
