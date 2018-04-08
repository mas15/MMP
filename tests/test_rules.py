import unittest
from unittest import mock
from unittest.mock import create_autospec
from parameterized import parameterized
from markets.rules import  extract_rules_to_file
import pandas as pd


# class TestRuleLearning(unittest.TestCase)

    #
    # def test_get_most_coefficient_features(self):
    #     res = self.pred_model.get_most_coefficient_features()
    #     features_sorted_by_coef = dict({"Up": [('F1', 1), ('F2', 2), ('F3', 3), ('F4', 4), ('F5', 5)],
    #                                     "Down": [('F5', 1), ('F4', 2), ('F3', 3), ('F2', 4), ('F1', 5)],
    #                                     "NC": [('F1', 10), ('F2', 11), ('F3', 12), ('F4', 13), ('F5', 14)]})
    #     self.assertEqual(features_sorted_by_coef, res)
