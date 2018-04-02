import unittest
from unittest import mock
from parameterized import parameterized
from markets.main_model_build import put_results_in_dict, get_misclassified_on_set, get_indexes_before_splitting, \
    sort_misclassified, MarketPredictingModel
import numpy as np
import pandas as pd


class TestMainModel(unittest.TestCase):

    def setUp(self):
        self.pred_model = MarketPredictingModel(MockModel(), MockExtractor())

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

    def test_put_results_in_dict(self):
        features = pd.DataFrame({'f1': [0], 'f2': [1], 'f3': [0], 'f4': [1], "Tweet_sentiment": [0.23]})
        propabilities = [("Down", 0.1), ("NC", 0.5), ("UP", 0.2)]
        prediction = "NC"
        res = put_results_in_dict(prediction, propabilities, features)
        exp_res = {'Down': 0.1, 'NC': 0.5, 'UP': 0.2, 'prediction': 'NC', 'features': ['f2', 'f4'],
                   'sentiment': 'Negative'}
        self.assertEqual(exp_res, res)

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


class MockExtractor:
    features = ["F1", "F2", "F3", "F4", "F5"]


class MockModel:
    classes_ = ["Up", "Down", "NC"]
    coef_ = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [10, 11, 12, 13, 14]]


if __name__ == '__main__':
    unittest.main()
