import unittest
from unittest import mock
from unittest.mock import create_autospec
from parameterized import parameterized
from markets.features_extractor import TweetFeaturesExtractor
from markets.market_predicting_model import put_results_in_dict, get_misclassified_on_set, get_indexes_before_splitting, \
    sort_misclassified, MarketPredictingModel
import numpy as np
from markets.helpers import k_split
from sklearn.naive_bayes import MultinomialNB
import pandas as pd


class TestMarketPredictingModel(unittest.TestCase):

    def setUp(self):
        features = ["F1", "F2", "F3", "F4", "F5"]

        mock_model = create_autospec(MultinomialNB)
        mock_model.classes_ = ["Up", "Down", "NC"]
        mock_model.coef_ = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [10, 11, 12, 13, 14]]

        mock_model.predict.return_value = np.array(["NC"])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.2, 0.5]])

        self.pred_model = MarketPredictingModel(features, mock_model)

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

    def test_train(self):  # todo missclassified?
        df = pd.DataFrame({"Text": ["Dummy", "Frame"], "Feature": [1, 0], "Target": [0, 1]})

        mock_split_sets = iter([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
        mock_accuracies = [10, 20, 12, 25, 19, 11, 15, 28]

        with mock.patch("sklearn.metrics.accuracy_score", side_effect=mock_accuracies, autospec=True) as _:
            with mock.patch("markets.helpers.k_split", return_value=mock_split_sets, autospec=True) as _:
                res = self.pred_model.train(df, k_folds=4)

                self.assertEqual((14.0, 21.0), res)
                self.assertEqual(4, self.pred_model.model.fit.call_count)


    # mock_extr = create_autospec(TweetFeaturesExtractor)
    # def test_analyse(self): # todo kiedys
    #     res = self.pred_model.analyse("Tweet content")
    #     expected_res = {'Down': 0.2, 'NC': 0.5, 'Up': 0.3,
    #                     'features': [],
    #                     'prediction': 'NC',
    #                     'sentiment': 'Positive'}
    #     self.assertEqual(expected_res, res)

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