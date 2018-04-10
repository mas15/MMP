import unittest
from unittest import mock
from unittest.mock import create_autospec
from parameterized import parameterized
from markets.predicting_model_trainer import ModelTrainer, ModelTrainingResult
from markets.features_extractor import TweetFeaturesExtractor
import pandas as pd


class TestModelTrainingResult(unittest.TestCase):

    def setUp(self):
        self.a = ModelTrainingResult(None, 60, 70)
        self.b = ModelTrainingResult(None, 65, 75)
        self.a.base_rate_accuracy, self.a.features = 40, range(1000)
        self.b.base_rate_accuracy, self.b.features = 50, range(1200)

    def test_constructor(self):
        df = pd.DataFrame({"Market_change": [0, 1, 1, 1], "F1": 1, "F2": 1, "F3": 1, "F4": 1})
        result = ModelTrainingResult(None, 66, 99, df)
        self.assertEqual(0.75, result.base_rate_accuracy)
        self.assertEqual(['F1', 'F2', 'F3', 'F4'], result.features)
        self.assertEqual(4, result.nr_tweets)

    def test_comparision_by_diff_between_base_rate_and_accuracy(self):
        self.assertGreater(self.a, self.b)

    def test_comparision_by_diff_between_nr_features(self):
        self.b.base_rate_accuracy = 45
        self.assertGreater(self.b, self.a)

#
# class TestModelTrainer(unittest.TestCase):
#
#     def setUp(self):
#         mock_df_processor = create_autospec(TweetFeaturesExtractor)
#         self.trainer = ModelTrainer(mock_df_processor)
#
#     def test_train(self):
#         df = pd.DataFrame({"Text": ["First", "Second"], "Market_change": [0.2, 0.5]})
#         # todo
#         #self.assertEqual(expected_result, result_df.to_dict())
#
#
#     def test_get_features_iterator(self):
#         #todo

