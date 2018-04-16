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
