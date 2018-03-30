import unittest
from markets.feature_selection import *
import numpy as np
import pandas as pd


class TestFeatureSelection(unittest.TestCase):
    def setUp(self):
        df = pd.DataFrame({"Text": ["one", "two"], "Market_change": [0, 1]})
        df = df[["Text", "Market_change"]] # todo ogarnac to
        self.selector = FeatureSelector(df)

    def test_get_indexes_sorted_by_score(self):
        values = [100, 20, 50, 30, 0, 35, 55]
        sorted_indexes = get_indexes_sorted_by_score(values)
        expected_result = [4, 1, 3, 5, 2, 6, 0]
        self.assertEqual(expected_result, sorted_indexes)

    def test_select_features_cached(self):
        self.selector.sorted_features = ["first", "second", "third", "forth"]
        res = self.selector.select_k_best_features(2)
        self.assertEqual(["first", "second"], res)


if __name__ == '__main__':
    unittest.main()
