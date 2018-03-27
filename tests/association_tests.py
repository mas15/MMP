from markets.association import get_date_to_check_affect, mark_features, drop_instances_without_features, mark_row, \
    filter_features, drop_infrequent_features, move_column_to_the_end, set_currency_change
import pandas as pd
import unittest
from markets.feature_extractor import FeatureExtractor


class TestAssosiationLearning(unittest.TestCase):
    def test_get_date_to_check_affect(self):
        d = pd.Timestamp('2017-01-02T23:47')
        exp_res = pd.Timestamp('2017-01-03')
        res = get_date_to_check_affect(d)
        self.assertEqual(exp_res, res)

    class MockFeatExtr:
        def extract_features(self, text):
            res = {"a": {"One": 1}, "b": {"Two": 1, "One": 1}, "c": {}}  # co jak np "Another": 1
            return res[text]

    def test_mark_row(self):
        extr = self.MockFeatExtr()
        df = pd.DataFrame({"Text": ['a', 'b', 'c'], "One": [0, 0, 0], "Two": [0, 0, 0], "Three": [0, 0, 0]})
        result = df.apply(lambda x: mark_row(x, extr), axis=1)
        exp_df = pd.DataFrame({"Text": ['a', 'b', 'c'], "One": [1, 1, 0], "Two": [0, 1, 0], "Three": [0, 0, 0]})
        self.assertTrue(exp_df.equals(result))

    def test_drop_instances_without_features(self):
        df = pd.DataFrame(
            {'Tweet_sentiment': [1, 1, 1], 'f1': [0, 0, 0], 'f2': [1, 1, 0], 'f3': [0, 1, 0],
             'Market_change': [0.1, 0.2, 0.3]})
        exp_df = pd.DataFrame(
            {'Tweet_sentiment': [1, 1], 'f1': [0, 0], 'f2': [1, 1], 'f3': [0, 1], 'Market_change': [0.1, 0.2]})
        result = drop_instances_without_features(df)
        self.assertTrue(exp_df.equals(result.reset_index(drop=True)))

    def test_filter_features(self):
        df = pd.DataFrame(columns=["Tweet_sentiment", "Text", 'a', 'b', 'c', 'd', 'e', "Market_change"])
        res = filter_features(df, ['b', 'd'])
        expected_columns = ["Tweet_sentiment", "Text", 'b', 'd', "Market_change"]
        self.assertEqual(expected_columns, res.columns.tolist())
        pass

    def test_drop_infrequent_features(self):
        df = pd.DataFrame({'Market_change': 1, 'Tweet_sentiment': 1, 'Text': 1,
                           '2_times': [0, 0, 0, 1, 1],
                           '4_times': [1, 1, 0, 1, 1],
                           '0_times': [0, 0, 0, 0, 0],
                           '5_times': [1, 1, 1, 1, 1]})
        expected_columns = sorted(["Market_change", 'Tweet_sentiment', 'Text', "4_times", "5_times"])
        res = drop_infrequent_features(df, min_freq=3)
        self.assertEqual(expected_columns, sorted(res.columns.tolist()))

    def test_move_column_to_the_end(self):
        df = pd.DataFrame(columns=["A", "B", "C", "D"])
        res = move_column_to_the_end(df, "B")
        self.assertEqual(["A", "C", "D", "B"], res.columns.tolist())

    def test_set_currency_change(self):
        df = pd.DataFrame({'Market_change': [1, 2, 3, 4, 5, 6, 7]})
        expected_changes = ["Down", "Down", "Down", "NC", "Up", "Up", "Up"]
        res = set_currency_change(df)
        self.assertEqual(expected_changes, res['Market_change'].values.tolist())

    def test_mark_features(self):
        extr = FeatureExtractor()
        features = "aaa bbb ccc ddd eee fff".split()
        extr.set_features(features)
        df = pd.DataFrame({'Text': ["aaa ccc eee", "ccc eee fff"]})

        res = mark_features(extr, df)
        res = res[sorted(res.columns)]
        exp_res = [['aaa ccc eee', 1, 0, 1, 0, 1, 0], ['ccc eee fff', 0, 0, 1, 0, 1, 1]]
        self.assertEqual(exp_res, res.values.tolist())


if __name__ == '__main__':
    unittest.main()
