import pandas as pd
from markets.dataset import TweetsDataSet, count_nr_of_feature_occurrences
from markets.sentiment import SentimentAnalyser
import unittest
from unittest import mock
from unittest.mock import create_autospec
from parameterized import parameterized


class TestTweetsDataSet(unittest.TestCase):
    def setUp(self):
        df = pd.DataFrame({"Text": ["First", "Second", "No features tweet"],
                           "F1": [0, 0, 0],
                           "F2": [1, 1, 0],
                           "F3": [1, 0, 0],
                           'Tweet_sentiment': [0.3, 0.6, 0.9],
                           "Market_change": [0.2, 0.5, 0.9]})
        self.dataset = TweetsDataSet(df)

    def test_features(self):
        self.assertEqual(["F1", "F2", "F3"], self.dataset.features)

    def test_get_x_y(self):
        x, y = self.dataset.get_x_y()
        self.assertEqual([[0, 1, 1, 0.3], [0, 1, 0, 0.6], [0, 0, 0, 0.9]], x.tolist())
        self.assertEqual([0.2, 0.5, 0.9], y.tolist())

    def test_get_all_tweets(self):
        self.assertEqual(['First', 'Second', 'No features tweet'], self.dataset.get_all_tweets())

    def test_get_features_only_df(self):
        expected_df = pd.DataFrame({"F1": [0, 0, 0], "F2": [1, 1, 0], "F3": [1, 0, 0], })
        self.assertEqual(expected_df.to_dict(), self.dataset.get_features_df().to_dict())

    def test_drop_instances_without_features(self):
        self.dataset.drop_instances_without_features()
        self.assertEqual(['First', 'Second'], self.dataset.get_all_tweets())

    def test_remove_features(self):
        self.dataset.remove_features(["F2"])
        self.assertEqual(["F1", "F3"], self.dataset.features)

    def test_remove_features_raises_if_feature_not_in_dataset(self):
        with self.assertRaises(Exception):  # todo test message
            self.dataset.remove_features(["F999"])

    def test_get_feature_occurencies(self):
        res = self.dataset.get_feature_occurencies()
        self.assertEqual([('F1', 0), ('F2', 2), ('F3', 1)], res)

    def test_set_sentiment(self):
        mock_sent_analyser = create_autospec(SentimentAnalyser)
        mock_sent_analyser.predict_score.side_effect = [0.3, 0.4, 0.5]
        self.dataset.df.drop(columns="Tweet_sentiment", inplace=True)

        self.dataset.set_sentiment(mock_sent_analyser.predict_score)
        self.assertEqual({0: 0.3, 1: 0.4, 2: 0.5}, self.dataset.df.to_dict()["Tweet_sentiment"])

    def test_mark_features(self):
        extracted = {"First": {"F1": 0, "F2": 1, "F3": 1},
                     "Second": {"F1": 0, "F2": 0, "F3": 0},
                     "No features tweet": {"F1": 1, "F2": 1, "F3": 1}}

        def extract_features(text):
            return extracted[text]

        self.dataset.df.drop(columns=["F1", "F2", "F3"], inplace=True)
        self.dataset.set_phrase_features(extract_features)
        x, _ = self.dataset.get_x_y()
        self.assertEqual([[0, 1, 1, 0.3], [0, 0, 0, 0.6], [1, 1, 1, 0.9]], x.tolist())

    def test_count_nr_of_feature_occurrences(self):
        df = pd.DataFrame({'2_times': [0, 0, 0, 1, 1],
                           '4_times': [1, 1, 0, 1, 1],
                           '0_times': [0, 0, 0, 0, 0],
                           '5_times': [1, 1, 1, 1, 1]})
        res = count_nr_of_feature_occurrences(df)
        self.assertEqual([('0_times', 0), ('2_times', 2), ('4_times', 4), ('5_times', 5)], res)


    # def test_mark_row(self):
    #     df = pd.DataFrame({"Text": ['a', 'b', 'c'], "A": [0, 0, 0], "B": [0, 0, 0], "C": [0, 0, 0]})
    #     result = df.apply(lambda x: self.dataset.mark_row(x, self.mock_extr), axis=1)
    #     exp_df = pd.DataFrame({"Text": ['a', 'b', 'c'], "A": [1, 1, 0], "B": [0, 1, 0], "C": [0, 0, 0]})
    #     self.assertTrue(exp_df.equals(result))
    #
