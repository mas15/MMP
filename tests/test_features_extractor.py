import unittest
from unittest import mock
import pandas as pd
from unittest.mock import create_autospec
from markets.features_extractor import TweetFeaturesExtractor
from markets.phrases_extractor import PhrasesExtractor
from markets.sentiment import SentimentAnalyser


class TestTweetFeaturesExtractor(unittest.TestCase):
    def setUp(self):
        mock_sent = create_autospec(SentimentAnalyser)
        mock_sent.predict_score.side_effect = lambda text: 0.3 if text == "First" else 0.6

        mock_extr = create_autospec(PhrasesExtractor)
        mock_extr.features = ["F1", "F2"]
        features = {"First": {"F1": 1, "F2": 0}, "Second": {"F1": 0, "F2": 1}, "No features tweet": {"F1": 0, "F2": 0}}
        mock_extr.extract_features.side_effect = lambda t: features[t]

        self.processor = TweetFeaturesExtractor(None, mock_extr, mock_sent)

    def test_extract_features(self):  # todo raise if no text, market_change
        df = pd.DataFrame({"Text": ["First", "Second"], "Market_change": [0.2, 0.5]})
        result_df = self.processor.extract_features(df)
        expected_result = {"Text": {0: "First", 1: "Second"},
                           "F1": {0: 1, 1: 0},
                           "F2": {0: 0, 1: 1},
                           'Tweet_sentiment': {0: 0.3, 1: 0.6},
                           "Market_change": {0: 0.2, 1: 0.5}}
        self.assertEqual(expected_result, result_df.to_dict())

    def test_filter_features(self):
        df = pd.DataFrame({"Text": ["First", "Second", "No features tweet"],
                           "F1": [0, 0, 0],
                           "F2": [1, 1, 0],
                           "F3": [1, 0, 1],
                           'Tweet_sentiment': [0.3, 0.6, 0.9],
                           "Market_change": [0.2, 0.5, 0.9]})
        result_df = self.processor.filter_features(df, ["F1", "F2"])

        expected_result = {"Text": {0: "First", 1: "Second"},
                           "F1": {0: 1, 1: 0},
                           "F2": {0: 0, 1: 1},
                           'Tweet_sentiment': {0: 0.3, 1: 0.6},
                           "Market_change": {0: 0.2, 1: 0.5}}
        self.assertEqual(expected_result, result_df.to_dict())
        self.processor.extr.set_features.assert_called_once_with(["F1", "F2"])

    def test_process_text(self):
        result_df = self.processor.process_text("First")  # czy dobrze ze nie ma textu w resulcie?
        expected_result = {"F1": {0: 1}, "F2": {0: 0}, 'Tweet_sentiment': {0: 0.3}}
        self.assertEqual(expected_result, result_df.to_dict())