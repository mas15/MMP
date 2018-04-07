import unittest
from unittest import mock
from unittest.mock import create_autospec
from parameterized import parameterized
from markets.main_model import put_results_in_dict, get_misclassified_on_set, get_indexes_before_splitting, \
    sort_misclassified, MarketPredictingModel, AssociationDataProcessor, ProvisionalPredictingModel
import numpy as np
from markets.helpers import k_split
from markets.feature_extractor import FeatureExtractor
from markets.sentiment import SentimentAnalyser
from sklearn.naive_bayes import MultinomialNB
import sklearn
import pandas as pd


class TestMainModel(unittest.TestCase):

    def setUp(self):
        # mock_extr = create_autospec(FeatureExtractor)
        # mock_extr.features = ["F1", "F2", "F3", "F4", "F5"]
        features = ["F1", "F2", "F3", "F4", "F5"]

        mock_model = create_autospec(MultinomialNB)
        mock_model.classes_ = ["Up", "Down", "NC"]
        mock_model.coef_ = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [10, 11, 12, 13, 14]]

        mock_model.predict.return_value = np.array(["NC"])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.2, 0.5]])
        self.pred_model = MarketPredictingModel("ABC", features, mock_model)

    # TODO te nie uzywane

    # def test_get_misclassified_objects(self):
    #     y = np.array(["Up", "NC", "Down", "Up"])
    #     predicted = np.array(["NC", "NC", "Down", "Down"])
    #     result = get_misclassified_on_set(y, predicted)
    #     self.assertEqual([0, 3], result[0].tolist())  # mozoe niech zwaraca z [0]?

    # @parameterized.expand([
    #     (np.array([0, 1, 3, 5, 6, 7, 8]), np.array([2, 3, 6]), [3, 5, 8]),
    #     (np.array([2, 4, 9]), np.array([1]), [4]),
    # ])
    # def test_get_indexes_before_splitting(self, indexes, misclassified, exp_result):
    #     res = get_indexes_before_splitting(indexes, misclassified)
    #     self.assertEqual(exp_result, res.tolist())

    # def test_sort_misclassified(self):
    #     misclassified_objects = dict([(1, 12), (2, 0), (3, 343), (4, 1), (5, 100)])
    #     res = sort_misclassified(misclassified_objects)
    #     exp_res = [(3, 343), (5, 100), (1, 12), (4, 1)]
    #     self.assertEqual(exp_res, res)

    def test_analyse(self):
        res = self.pred_model.analyse("Tweet content")
        expected_res = {'Down': 0.2, 'NC': 0.5, 'Up': 0.3,
                        'features': [],  # czy tu pusto?
                        'prediction': 'NC',
                        'sentiment': 'Positive'}
        self.assertEqual(expected_res, res)

    def test_get_most_coefficient_features(self):
        res = self.pred_model.get_most_coefficient_features()
        features_sorted_by_coef = dict({"Up": [('F1', 1), ('F2', 2), ('F3', 3), ('F4', 4), ('F5', 5)],
                                        "Down": [('F5', 1), ('F4', 2), ('F3', 3), ('F2', 4), ('F1', 5)],
                                        "NC": [('F1', 10), ('F2', 11), ('F3', 12), ('F4', 13), ('F5', 14)]})
        self.assertEqual(features_sorted_by_coef, res)

    def test_get_most_coefficient_features_raises_exception(self):
        self.pred_model.model.classes_ = ["Just one"]
        with self.assertRaises(Exception):
            self.pred_model.get_most_coefficient_features()


class TestAssociationDataProcessor(unittest.TestCase):
    def setUp(self):
        mock_sent = create_autospec(SentimentAnalyser)
        mock_sent.predict_score.side_effect = lambda text: 0.3 if text == "First" else 0.6

        mock_extr = create_autospec(FeatureExtractor)
        mock_extr.features = ["F1", "F2"]
        features = {"First": {"F1": 1, "F2": 0}, "Second": {"F1": 0, "F2": 1}, "No features tweet": {"F1": 0, "F2": 0}}
        mock_extr.extract_features.side_effect = lambda t: features[t]

        self.processor = AssociationDataProcessor(None, mock_extr, mock_sent)

    def test_extract_features(self):  # todo raise if no text, market_change
        df = pd.DataFrame({"Text": ["First", "Second"], "Market_change": [0.2, 0.5]})
        result_df = self.processor.extract_features(df)
        expected_result = {"Text": {0: "First", 1: "Second"},
                           "F1": {0: 1, 1: 0},
                           "F2": {0: 0, 1: 1},
                           'Tweet_sentiment': {0: 0.3, 1: 0.6},
                           "Market_change": {0: 0.2, 1: 0.5}}
        self.assertEqual(expected_result, result_df.to_dict())

    def test_filter_features(self):
        df = pd.DataFrame({"Text": ["First", "Second", "No features tweet"],
                           "F1": [0, 0, 0],
                           "F2": [1, 1, 0],
                           "F3": [1, 0, 1],
                           'Tweet_sentiment': [0.3, 0.6, 0.9],
                           "Market_change": [0.2, 0.5, 0.9]})
        result_df = self.processor.filter_features(df, ["F1", "F2"])

        expected_result = {"Text": {0: "First", 1: "Second"},
                           "F1": {0: 1, 1: 0},
                           "F2": {0: 0, 1: 1},
                           'Tweet_sentiment': {0: 0.3, 1: 0.6},
                           "Market_change": {0: 0.2, 1: 0.5}}
        self.assertEqual(expected_result, result_df.to_dict())
        self.processor.extr.set_features.assert_called_once_with(["F1", "F2"])

    def test_process_text(self):
        result_df = self.processor.process_text("First")  # czy dobrze ze nie ma textu w resulcie?
        expected_result = {"F1": {0: 1}, "F2": {0: 0}, 'Tweet_sentiment': {0: 0.3}}
        self.assertEqual(expected_result, result_df.to_dict())


class TestProvisionalPredictingModel(unittest.TestCase):
    def setUp(self):
        mock_model = create_autospec(MultinomialNB)
        mock_model.classes_ = ["Up", "Down", "NC"]
        mock_model.coef_ = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [10, 11, 12, 13, 14]]
        self.prov_model = ProvisionalPredictingModel(mock_model)

    def test_train(self):  # todo missclassified?
        df = pd.DataFrame({"Text": ["Dummy", "Frame"], "Feature": [1, 0], "Target": [0, 1]})

        mock_split_sets = iter([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
        mock_accuracies = [10, 20, 12, 25, 19, 11, 15, 28]

        with mock.patch("sklearn.metrics.accuracy_score", side_effect=mock_accuracies, autospec=True) as _:
            with mock.patch("markets.helpers.k_split", return_value=mock_split_sets, autospec=True) as _:
                res = self.prov_model.train(df, k_folds=4)

                self.assertEqual((14.0, 21.0), res)
                self.assertEqual(4, self.prov_model.model.fit.call_count)


class TestOtherFunctions(unittest.TestCase):
    def test_put_results_in_dict(self):
        features = pd.DataFrame({'f1': [0], 'f2': [1], 'f3': [0], 'f4': [1], "Tweet_sentiment": [0.23]})
        propabilities = [("Down", 0.1), ("NC", 0.5), ("UP", 0.2)]
        prediction = "NC"
        res = put_results_in_dict(prediction, propabilities, features)
        exp_res = {'Down': 0.1, 'NC': 0.5, 'UP': 0.2, 'prediction': 'NC', 'features': ['f2', 'f4'],
                   'sentiment': 'Negative'}
        self.assertEqual(exp_res, res)


if __name__ == '__main__':
    unittest.main()
