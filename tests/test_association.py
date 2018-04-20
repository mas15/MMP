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

    # def test_set_currency_change(self):
    #     dataset = create_autospec(TweetsDataSet)
    #     dataset.get_market_change.return_value = [1, 2, 3, 4, 5, 6, 7]
    #
    #     set_currency_change(dataset)
    #
    #     expected_changes = ["Down", "Down", "Down", "NC", "Up", "Up", "Up"]
    #     dataset.set_market_change.assert_called_once_with(expected_changes)

    def test_calculate_thresholds(self):
        market_change = pd.Series([1, 2, 3, 4, 5, 6, 7])
        lower, higher = calculate_thresholds(market_change)
        self.assertEqual((3.33, 4.67), (lower, higher))

    # def test_save_sifted_tweets_with_date(self):
    #     input_df = pd.DataFrame({"Text": ["First", "Second"], "F1": [0, 0], "F2": [1, 1]})
    #     mock_curr_values = pd.DataFrame({"Date": ['Mar 06, 2018', 'Mar 07, 2018', 'Mar 08, 2018'],
    # "Change": [1.3, 2.6, 0.7], "Price": 0, "Open": 0, "High": 0, "Low": 0, "Vol.": 0})
    #
    #     mock_tweets = pd.DataFrame({"Text": ["First", "Second", "Third"],
    #                                 "Date": ["2018-03-06 11:22:33", "2018-03-07 22:33:44", "2018-03-05 12:57:12"],
    #                                 "Id": [1, 2, 3]})
    #     mock_tweets['Date'] = pd.to_datetime(mock_tweets['Date'], format='%Y-%m-%d %H:%M:%S')
    #
    #     expected_result = pd.DataFrame({"Text": ["First", "Second"],
    #                                     'Date': ['2018-03-06', '2018-03-08'],
    #                                     "Market_change": [1.3, 0.7]})
    #
    #     with mock.patch("pandas.read_csv", return_value=mock_curr_values):
    #         with mock.patch("markets.association.read_all_tweets", return_value=mock_tweets):
    #             with mock.patch("pandas.core.frame.DataFrame.to_csv"):
    #                 res = save_sifted_tweets_with_date(input_df, "filename")
    #                 self.assertEqual(expected_result.to_dict(), res.to_dict())

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
