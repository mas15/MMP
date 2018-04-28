import pandas as pd
import numpy as np
import unittest
from unittest import mock
from unittest.mock import patch, create_autospec
from markets.dataset import TweetsDataSet
from parameterized import parameterized
from markets.association import set_currency_change, calculate_thresholds, \
    save_sifted_tweets_with_date, read_currency_prices


class TestTweetsAssociationWithMarkets(unittest.TestCase):

    def test_calculate_thresholds(self):
        market_change = pd.Series([1, 2, 3, 4, 5, 6, 7])
        lower, higher = calculate_thresholds(market_change)
        self.assertEqual((3.33, 4.67), (lower, higher))

    @parameterized.expand([(True,), (False,)])
    def test_read_currency_prices(self, additional_cols):
        mock_data = {"Date": ['Mar 06, 2018', 'Mar 07, 2018', 'Mar 08, 2018'], "Change": [1.3, 2.6, 0.7],
                     "Open": [110, 220, 330]}
        if additional_cols:
            mock_data.update({"Price": 0, "High": 0, "Low": 0, "Vol.": 0})
        with mock.patch("pandas.read_csv", return_value=pd.DataFrame(mock_data)):
            res = read_currency_prices("ABC")
            self.assertEqual(['Date', 'Open', 'Market_change'], res.columns.tolist())
            self.assertEqual(np.dtype('datetime64[ns]'), res["Date"].dtype)
            self.assertEqual([1.3, 2.6, 0.7], res["Market_change"].tolist())
            self.assertEqual([110, 220, 330], res["Open"].tolist())


if __name__ == '__main__':
    unittest.main()
