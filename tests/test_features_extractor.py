import unittest
from unittest import mock
import pandas as pd
from unittest.mock import create_autospec
from markets.tweets_features_extraction import FeatureExtractor
from markets.phrases_extractor import PhrasesExtractor
from markets.sentiment import SentimentAnalyser
from markets.dataset import TweetsDataSet


class TestTweetFeaturesExtractor(unittest.TestCase):
    def setUp(self):
        mock_sent = create_autospec(SentimentAnalyser)
        mock_sent.predict_score.side_effect = lambda text: 0.3 if text == "First" else 0.6

        mock_extr = create_autospec(PhrasesExtractor)
        mock_extr.features = ["F1", "F2"] # todo usunac ta lnijke
        features = {"First": {"F1": 1, "F2": 0}, "Second": {"F1": 0, "F2": 1}, "No features tweet": {"F1": 0, "F2": 0}}
        mock_extr.extract_features.side_effect = lambda t: features[t]

        dataset = TweetsDataSet(pd.DataFrame({"Text": ["First", "Second", "No features tweet"], "Market_change": [0.2, 0.5, 0.9]}))
        self.extractor = FeatureExtractor(dataset, None, mock_extr, mock_sent, 1)

    def test_extract_features(self):
        result = self.extractor.extract_features()
        expected_result = {"Text": {0: "First", 1: "Second"},
                           "F1": {0: 1, 1: 0},
                           "F2": {0: 0, 1: 1},
                           'Tweet_sentiment': {0: 0.3, 1: 0.6},
                           "Market_change": {0: 0.2, 1: 0.5}}
        self.assertEqual(expected_result, result.df.to_dict())

    #
    # def test_filter_features(self):
    #     df = pd.DataFrame({"Text": ["First", "Second", "No features tweet"],
    #                        "F1": [0, 0, 0],
    #                        "F2": [1, 1, 0],
    #                        "F3": [1, 0, 1],
    #                        'Tweet_sentiment': [0.3, 0.6, 0.9],
    #                        "Market_change": [0.2, 0.5, 0.9]})
    #     result_df = self.processor.filter_features(df, ["F1", "F2"])
    #
    #     expected_result = {"Text": {0: "First", 1: "Second"},
    #                        "F1": {0: 1, 1: 0},
    #                        "F2": {0: 0, 1: 1},
    #                        'Tweet_sentiment': {0: 0.3, 1: 0.6},
    #                        "Market_change": {0: 0.2, 1: 0.5}}
    #     self.assertEqual(expected_result, result_df.to_dict())
    #     self.processor.extr.set_features.assert_called_once_with(["F1", "F2"])
    #
    # def test_process_text(self):
    #     result_df = self.processor.process_text("First")  # czy dobrze ze nie ma textu w resulcie?
    #     expected_result = {"F1": {0: 1}, "F2": {0: 0}, 'Tweet_sentiment': {0: 0.3}}
    #     self.assertEqual(expected_result, result_df.to_dict())

    #
    # @parameterized.expand([(3, []), (2, ["F2"]), (1, ["F2", "F3"])])
    # def test_get_feature_occurencies(self, min_occ, exp_res):
    #     freq = self.dataset.get_infrequent_features(min_occ)
    #     self.assertEqual(exp_res, freq)