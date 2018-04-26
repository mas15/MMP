import unittest
from unittest import mock
from unittest.mock import create_autospec
from parameterized import parameterized
from markets.market_predicting_model import AnalysisResult, Classifier, format_result, MarketPredictingModel
from markets.utils import k_split
from markets.dataset import TweetsDataSet
import pandas as pd


class TestMarketPredictingModel(unittest.TestCase):

    def setUp(self):
        main_model = create_autospec(Classifier)
        rest_model = create_autospec(Classifier)
        main_model.analyse.return_value = {"Down": 0.1, "NC": 0.5, "Up": 0.2}
        rest_model.analyse.return_value = {"Down": 0.9, "NC": 0.1, "Up": 0.4}
        self.pred_model = MarketPredictingModel(main_model, rest_model)
        self.pred_model.main_features = ["A", "B", "C"]
        self.pred_model.all_features = ["A", "B", "C", "D", "E"]

        self.tweet_dataset = create_autospec(TweetsDataSet)
        self.sifted_dataset = create_autospec(TweetsDataSet)
        self.tweet_dataset.get_sentiment.return_value = [0.2]
        self.sifted_dataset.get_sentiment.return_value = [0.8]

    def test_analyse_tweet_with_main_features(self):
        self.tweet_dataset.get_marked_features.return_value = ["B", "C"]
        self.sifted_dataset.get_marked_features.return_value = ["B", "C"]

        result = self.pred_model.analyse(self.tweet_dataset, self.sifted_dataset)

        expected_result = {"Sentiment": "Positive", "Features": "B, C",
                           "Down": 0.1, "NC": 0.5, "Up": 0.2, 'Prediction': 'No change'}
        self.assertEqual(expected_result, result.to_dict())

    def test_analyse_tweet_with_rest_features(self):
        self.tweet_dataset.get_marked_features.return_value = ["D", "E"]
        self.sifted_dataset.get_marked_features.return_value = []

        result = self.pred_model.analyse(self.tweet_dataset, self.sifted_dataset)

        expected_result = {"Sentiment": "Negative", "Features": "D, E",
                           "Down": 0.9, "NC": 0.1, "Up": 0.4, 'Prediction': 'Down'}
        self.assertEqual(expected_result, result.to_dict())

    def test_analyse_tweet_with_no_features(self):
        self.tweet_dataset.get_marked_features.return_value = []
        self.sifted_dataset.get_marked_features.return_value = []

        result = self.pred_model.analyse(self.tweet_dataset, self.sifted_dataset)

        expected_result = {"Sentiment": "Negative", "Features": 'No features found in the tweet',
                           "Down": 0.5, "NC": 0.3, "Up": 0.3, 'Prediction': 'Down'}
        self.assertEqual(expected_result, result.to_dict())

    def test_get_most_coefficient_features(self):
        res = self.pred_model.main_model.get_most_coefficient_features()
        # TODO


class TestAnalysisResult(unittest.TestCase):
    def setUp(self):
        self.probabilities = dict([("Down", 0.1), ("NC", 0.5), ("Up", 0.2)])
        self.result = AnalysisResult(self.probabilities, 0.34, ["f1", "f2", "f3"])

    def test_constructor(self):
        self.assertEqual("NC", self.result.prediction)

    def test_to_dict(self):
        expected_result = {"Sentiment": "Negative", "Features": "f1, f2, f3",
                           "Down": 0.1, "NC": 0.5, "Up": 0.2, 'Prediction': 'No change'}
        self.assertEqual(expected_result, self.result.to_dict())

    def test_to_dict_when_no_features(self):
        self.result.features = []
        self.assertEqual("No features found in the tweet", self.result.to_dict()["Features"])

    def test_combine_with(self):
        other = AnalysisResult({"Down": 0.9, "NC": 0.0, "Up": 0.4}, 0.76, [])
        self.result.combine_with(other)
        expected_result = {"Sentiment": "Positive", "Features": "f1, f2, f3",
                           "Down": 0.5, "NC": 0.25, "Up": 0.3, 'Prediction': 'Down'}
        self.assertEqual(expected_result, self.result.to_dict())

    def test_format_result(self):
        dataset = create_autospec(TweetsDataSet)
        dataset.get_sentiment.return_value = [0.2]
        dataset.get_marked_features.return_value = ["f1", "f2", "f3"]
        result = format_result(self.probabilities, dataset)
        expected_result = {"Sentiment": "Negative", "Features": "f1, f2, f3",
                           "Down": 0.1, "NC": 0.5, "Up": 0.2, 'Prediction': 'No change'}
        self.assertEqual(expected_result, result.to_dict())


if __name__ == '__main__':
    unittest.main()
