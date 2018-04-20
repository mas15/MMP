import unittest
from unittest import mock
import pandas as pd
from parameterized import parameterized
from unittest.mock import create_autospec
from markets.tweets_features_extraction import build_dataset_with_one_tweet, extract_features, remark_features
from markets.phrases_extractor import PhrasesExtractor
from markets.sentiment import SentimentAnalyser
from markets.dataset import TweetsDataSet


class TestTweetFeaturesExtractor(unittest.TestCase):
    def setUp(self):
        self.mock_sent = create_autospec(SentimentAnalyser)
        self.mock_sent.predict_score.side_effect = lambda text: 0.3 if text == "First" else 0.6

        self.mock_extr = create_autospec(PhrasesExtractor)
        self.mock_extr.features = ["F1", "F2"]
        features = {"First": {"F1": 1, "F2": 0}, "Second": {"F1": 0, "F2": 1}, "No features tweet": {"F1": 0, "F2": 0}}
        self.mock_extr.extract_features.side_effect = lambda t: features[t]

    def test_extract_features(self):
        with mock.patch("markets.tweets_features_extraction.PhrasesExtractor", return_value=self.mock_extr):
            with mock.patch("markets.tweets_features_extraction.SentimentAnalyser", return_value=self.mock_sent):
                dataset = TweetsDataSet(pd.DataFrame({"Text": ["First", "Second", "No features tweet"], "Market_change": [0.2, 0.5, 0.9]}))
                result = extract_features(dataset, ["F1", "F2"], True, 1)
                expected_result = {"Text": {0: "First", 1: "Second"},
                                   "F1": {0: 1, 1: 0},
                                   "F2": {0: 0, 1: 1},
                                   'Tweet_sentiment': {0: 0.3, 1: 0.6},
                                   "Market_change": {0: 0.2, 1: 0.5}}
                self.assertEqual(expected_result, result.df.to_dict())
                # TODO asset feature extractor called with set features etc

    def test_build_dataset_with_one_tweet(self):
            with mock.patch("markets.tweets_features_extraction.PhrasesExtractor", return_value=self.mock_extr):
                with mock.patch("markets.tweets_features_extraction.SentimentAnalyser", return_value=self.mock_sent):
                    dataset = build_dataset_with_one_tweet("First", ["F1", "F2"])
                    self.assertEqual(["First"], dataset.get_all_tweets())
                    self.assertEqual(["F1", "F2"], dataset.features)
                    self.assertEqual([[1, 0, 0.3]], dataset.get_x().tolist())
                # TODO asset feature extractor called with set features etc

    @parameterized.expand([(True,), (False,)])
    def test_remark_features(self, with_dropping):
        with mock.patch("markets.tweets_features_extraction.PhrasesExtractor", return_value=self.mock_extr):
            dataset = TweetsDataSet(pd.DataFrame({"Text": ["First", "Second", "No features tweet"],
                                                  "Market_change": [0.2, 0.5, 0.9]}))
            result = remark_features(dataset, ["F1", "F2"], with_dropping)
            if with_dropping:
                expected_result = {"Text": {0: "First", 1: "Second"},
                                   "F1": {0: 1, 1: 0},
                                   "F2": {0: 0, 1: 1},
                                   "Market_change": {0: 0.2, 1: 0.5}}
            else:
                expected_result = {"Text": {0: "First", 1: "Second", 2: "No features tweet"},
                                   "F1": {0: 1, 1: 0, 2: 0},
                                   "F2": {0: 0, 1: 1, 2: 0},
                                   "Market_change": {0: 0.2, 1: 0.5, 2: 0.9}}
            self.assertEqual(expected_result, result.df.to_dict())
