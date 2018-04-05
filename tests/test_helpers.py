import unittest
from unittest import mock
from unittest.mock import create_autospec
from markets.feature_extractor import FeatureExtractor
from markets.helpers import get_x_y_from_df, move_column_to_the_end, drop_instances_without_features, \
    remove_features, mark_features, mark_row, get_x_y_from_list_of_tuples, count_nr_of_feature_occurrences
import pandas as pd


class TestHelpers(unittest.TestCase):
    def setUp(self):
        features = ["A", "B", "C", "D"]
        extracted = {"a": {"A": 1}, "b": {"B": 1, "A": 1}, "c": {},
                     "first": dict(zip(features, [1, 0, 1, 0])), "second": dict(zip(features, [0, 1, 1, 0]))}

        def extract_features(text):
            return extracted[text]

        self.mock_extr = create_autospec(FeatureExtractor)
        self.mock_extr.features = features
        self.mock_extr.extract_features = extract_features

    def test_get_x_y_from_df(self):
        df = pd.DataFrame({"Text": ["one", "two"], "f1": [1, 0], "f2": [0, 1],
                           "Tweet_sentiment": [0.2, 0.5], "Target": ["Up", "Down"]})
        df = df[["Text", "f1", "f2", "Tweet_sentiment", "Target"]]  # todo tworzyc inaczej
        x, y = get_x_y_from_df(df)
        self.assertEqual([[1.0, 0.0, 0.2], [0.0, 1.0, 0.5]], x.tolist())
        self.assertEqual(["Up", "Down"], y.tolist())

    def test_get_x_y_from_list_of_tuples(self):
        x, y = [1, 2, 3], [1, 4, 9]
        res_x, res_y = get_x_y_from_list_of_tuples(zip(x, y))
        self.assertEqual((x, y), (res_x.tolist(), res_y.tolist()))

    def test_move_column_to_the_end(self):
        df = pd.DataFrame(columns=["A", "B", "C", "D"])
        res = move_column_to_the_end(df, "B")
        self.assertEqual(["A", "C", "D", "B"], res.columns.tolist())

    def test_drop_instances_without_features(self):
        df = pd.DataFrame(
            {'Text': ['a', 'b', 'c'], 'Tweet_sentiment': [1, 1, 1], 'f1': [0, 0, 0], 'f2': [1, 1, 0], 'f3': [0, 1, 0],
             'Market_change': [0.1, 0.2, 0.3]})
        exp_df = pd.DataFrame(
            {'Text': ['a', 'b'], 'Tweet_sentiment': [1, 1], 'f1': [0, 0], 'f2': [1, 1], 'f3': [0, 1],
             'Market_change': [0.1, 0.2]})  # todo drop last row
        result = drop_instances_without_features(df)
        self.assertTrue(exp_df.equals(result.reset_index(drop=True)))

    def test_remove_features(self):  # to teraz jebnie
        df = pd.DataFrame(columns=["Tweet_sentiment", "Text", 'a', 'b', 'c', 'd', 'e', "Market_change"])
        res = remove_features(df, ['b', 'd'])
        self.assertEqual(['b', 'd'], res.columns.tolist())
        pass

    def test_mark_row(self):
        df = pd.DataFrame({"Text": ['a', 'b', 'c'], "A": [0, 0, 0], "B": [0, 0, 0], "C": [0, 0, 0]})
        result = df.apply(lambda x: mark_row(x, self.mock_extr), axis=1)
        exp_df = pd.DataFrame({"Text": ['a', 'b', 'c'], "A": [1, 1, 0], "B": [0, 1, 0], "C": [0, 0, 0]})
        self.assertTrue(exp_df.equals(result))

    def test_mark_features(self):
        df = pd.DataFrame({'Text': ["first", "second"]})
        res = mark_features(self.mock_extr, df)
        exp_res = [["first", 1, 0, 1, 0], ["second", 0, 1, 1, 0]]
        self.assertEqual(exp_res, res.values.tolist())

    # todo inne do mark features

    def test_count_nr_of_feature_occurrences(self):
        df = pd.DataFrame({'2_times': [0, 0, 0, 1, 1],
                           '4_times': [1, 1, 0, 1, 1],
                           '0_times': [0, 0, 0, 0, 0],
                           '5_times': [1, 1, 1, 1, 1]})
        res = count_nr_of_feature_occurrences(df)
        self.assertEqual([('0_times', 0), ('2_times', 2), ('4_times', 4), ('5_times', 5)], res)


if __name__ == '__main__':
    unittest.main()
