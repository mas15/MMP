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
    #
    # def test_train(self):  # todo missclassified?
    #     df = pd.DataFrame({"Text": ["Dummy", "Frame"], "Feature": [1, 0], "Target": [0, 1]})
    #
    #     mock_split_sets = iter([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    #     mock_accuracies = [10, 20, 12, 25, 19, 11, 15, 28]
    #
    #     with mock.patch("sklearn.metrics.accuracy_score", side_effect=mock_accuracies, autospec=True) as _:
    #         with mock.patch("markets.helpers.k_split", return_value=mock_split_sets, autospec=True) as _:
    #             res = self.pred_model.train(df, k_folds=4)
    #
    #             self.assertEqual((14.0, 21.0), res)
    #             self.assertEqual(4, self.pred_model.model.fit.call_count)

    def test_get_most_coefficient_features(self):
        res = self.pred_model.get_most_coefficient_features()
        features_sorted_by_coef = dict({"Up": [('F1', 1), ('F2', 2), ('F3', 3), ('F4', 4), ('F5', 5)],
                                        "Down": [('F5', 1), ('F4', 2), ('F3', 3), ('F2', 4), ('F1', 5)],
                                        "NC": [('F1', 10), ('F2', 11), ('F3', 12), ('F4', 13), ('F5', 14)]})
        self.assertEqual(features_sorted_by_coef, res)

    def test_get_most_coefficient_features_raises_exception(self):
        self.pred_model.model.classes_ = ["Just one"]
        with self.assertRaises(Exception):
            self.pred_model.get_most_coefficient_features()


class TestAnalysisResult(unittest.TestCase):
    def setUp(self):
        probabilities = dict([("Down", 0.1), ("NC", 0.5), ("Up", 0.2)])
        self.result = AnalysisResult(probabilities, 0.34, ["f1", "f2", "f3"])

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
        pass # TODO

if __name__ == '__main__':
    unittest.main()
