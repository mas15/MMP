import unittest
from unittest import mock
from unittest.mock import create_autospec
from parameterized import parameterized
from markets.rules import get_rules_filename, extract_rules_to_file, get_rules_data
import pandas as pd


class TestRuleLearning(unittest.TestCase):

    def test_get_rules_filename(self):
        with mock.patch("markets.rules.DATA_PATH", "data_path_here"):
            res = get_rules_filename("ABC")
            self.assertEqual(r"data_path_here\ABC_rules.csv", res)

    def test_get_rules_data(self):
        mock_data = pd.DataFrame({"antecedants": ["travel ban, politically", "milk, bread"],
                                  "consequents": ["correct", "butter"],
                                  "support": [0.0047694753577106515, 0.00123456789],
                                  "lift": [78.625, 'inf']})
        with mock.patch("pandas.read_csv", return_value=mock_data):
            rules = get_rules_data("ABC")
            self.assertEqual(sorted(['antecedants', 'consequents', 'support', 'lift']), sorted(rules["columns"]))
            # self.assertEqual(['2018-03-06 11:22:33', '2018-03-07 22:33:44'], dates)
            # self.assertEqual({'2018-03-06 11:22:33': 'First', '2018-03-07 22:33:44': 'Second'}, tweets_per_date)


    #
    # def test_get_most_coefficient_features(self):
    #     res = self.pred_model.get_most_coefficient_features()
    #     features_sorted_by_coef = dict({"Up": [('F1', 1), ('F2', 2), ('F3', 3), ('F4', 4), ('F5', 5)],
    #                                     "Down": [('F5', 1), ('F4', 2), ('F3', 3), ('F2', 4), ('F1', 5)],
    #                                     "NC": [('F1', 10), ('F2', 11), ('F3', 12), ('F4', 13), ('F5', 14)]})
    #     self.assertEqual(features_sorted_by_coef, res)
