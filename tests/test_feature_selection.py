import unittest
from unittest import mock
from markets.feature_selection import *
import numpy as np
import pandas as pd
from parameterized import parameterized


class TestFeatureSelection(unittest.TestCase):
    def setUp(self):
        df = pd.DataFrame(columns=["Text", "Market_change", "F1", "F2", "F3", "F4", "F5", "F6"])
        df.loc[0] = ["text", "0.5", 1, 0, 1, 0, 1, 0]
        self.selector = FeatureSelector(df)

    def test_constructor(self):
        self.assertEqual(["F1", "F2", "F3", "F4", "F5", "F6"], self.selector.features_names)

    def test_select_features_cached(self):
        self.selector.sorted_features = ["first", "second", "third", "forth"]
        res = self.selector.select_k_best_features(2)
        self.assertEqual(["first", "second"], res)

    @parameterized.expand([
        (100, ['F3', 'F1', 'F4', 'F2', 'F5', 'F6']),
        (6, ['F3', 'F1', 'F4', 'F2', 'F5', 'F6']),
        (3, ['F3', 'F1', 'F4']),
        (0, []),
    ])
    def test_select_k_best_features(self, k, exp_res):
        with mock.patch('sklearn.feature_selection.RFECV') as m:
            instance = m.return_value
            instance.fit.return_value = instance
            instance.ranking_ = np.array([2, 4, 1, 3, 5, 11])

            res = self.selector.select_k_best_features(k)
            self.assertEqual(exp_res, res)

    def test_drop_infrequent_features(self):
        df = pd.DataFrame({'Market_change': 1, 'Tweet_sentiment': 1, 'Text': 1,
                           '2_times': [0, 0, 0, 1, 1],
                           '4_times': [1, 1, 0, 1, 1],
                           '0_times': [0, 0, 0, 0, 0],
                           '5_times': [1, 1, 1, 1, 1]})
        res = get_frequent_features(df, min_freq=3)
        self.assertEqual(["4_times", "5_times"], res)

    def test_get_indexes_sorted_by_score(self):
        values = [100, 20, 50, 30, 0, 35, 55]
        sorted_indexes = get_indexes_sorted_by_score(values)
        expected_result = [4, 1, 3, 5, 2, 6, 0]
        self.assertEqual(expected_result, sorted_indexes)


if __name__ == '__main__':
    unittest.main()
