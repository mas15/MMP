import os
import pandas as pd
from markets.main_model import AssociationDataProcessor, MarketPredictingModel, ProvisionalPredictingModel
from markets.feature_selection import get_frequent_features, get_best_features_from_file, get_k_best_features

pd.set_option('display.width', 1500)
pd.options.display.max_colwidth = 1000


class ModelTrainer:
    def __init__(self, df_processor=None):
        self.df_processor = df_processor or AssociationDataProcessor()

    def train(self, df, features_filename=None): # todo test
        best_features = self._find_best_features(df, features_filename)

        df_with_features = self.df_processor.extract_features(df)
        sifted_df = self.df_processor.filter_features(df_with_features, best_features)

        model = MarketPredictingModel(best_features)
        accuracies = model.train(sifted_df)

        return model, sifted_df, accuracies

    def _find_best_features(self, df, features_filename):
        best_accuracy, best_k, best_features = (0, 0), 0, []

        df = self.df_processor.extract_features(df)
        features = get_frequent_features(df)
        df = self.df_processor.filter_features(df, features)

        for features, k_features in get_features_iterator(df, features_filename):
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


def get_features_iterator(df, selected_features_filename=None):  # todo test
    if selected_features_filename:
        if os.path.isfile(selected_features_filename):
            return get_best_features_from_file(selected_features_filename)
        # lgo here no file
    return get_k_best_features(df, 100, 130)

