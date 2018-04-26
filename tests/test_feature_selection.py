import unittest
from unittest import mock
from unittest.mock import create_autospec
from markets.feature_selection import filter_features
from markets.dataset import TweetsDataSet
import numpy as np
import pandas as pd
from parameterized import parameterized


class TestFeatureSelection(unittest.TestCase):
    def setUp(self):
        df = pd.DataFrame({"Text": ["First", "Second", "No features tweet"],
                           "F1": [0, 0, 0],
                           "F2": [1, 1, 0],
                           "F3": [1, 0, 0],
                           'Tweet_sentiment': [0.3, 0.6, 0.9],
                           "Market_change": [0.2, 0.5, 0.9]})
        self.dataset = TweetsDataSet(df)

    def test_select_features(self):
        with mock.patch('markets.feature_selection.get_features_from_weka', autospec=True) as mock_get_from_weka:
            with mock.patch('builtins.open', autospec=True) as mock_open:
                mock_get_from_weka.return_value = ["F1", "F2"]

                dataset = create_autospec(TweetsDataSet)
                result = filter_features(dataset, ["F1", "F2"]) # TODO finish this test

                self.assertEqual(["F1", "F2"], result.features)

                mock_open.assert_called_with("path/to/open")
                handle = mock_open()
                handle.write.assert_called_once_with(["F1\nF2"])
