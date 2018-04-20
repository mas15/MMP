import unittest
from unittest import mock
from unittest.mock import create_autospec
from markets.feature_selection import *
import numpy as np
import pandas as pd
from parameterized import parameterized


class TestFeatureSelection(unittest.TestCase):

    def test_filter_features(self):
        with mock.patch('markets.feature_selection.PhrasesExtractor', autospec=True) as mock_extr:
            extr = mock_extr.return_value  # assert called
            dataset = create_autospec("markets.dataset.TweetsDataSet")

            result = filter_features(dataset, ["F1", "F2"])

            dataset.assert_called_once_with(["F1", "F2"])

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