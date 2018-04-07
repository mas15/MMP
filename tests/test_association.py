import pandas as pd
import numpy as np
import unittest
from unittest import mock
from unittest.mock import patch
from parameterized import parameterized
from markets.association import get_date_to_check_affect, set_currency_change, calculate_thresholds, \
    get_tweets_with_affect_filename, get_currency_prices_filename, get_graph_filename, get_graph_data, \
    save_sifted_tweets_with_date, read_currency_prices, set_date_with_effect
from markets import helpers


@patch("markets.association.DATA_PATH", "data_path_here")
class TestTweetsAssociationWithMarkets(unittest.TestCase):  # todo rename
    def setUp(self):
        self.mock_letters = mock.patch.object(
            helpers, 'DATA_PATH', return_value="data_path_here"
        )

    def test_get_date_to_check_affect(self):
        d = pd.Timestamp('2017-01-02T23:47')
        exp_res = pd.Timestamp('2017-01-03')
        res = get_date_to_check_affect(d)
        self.assertEqual(exp_res, res)

    def test_set_currency_change(self):
        df = pd.DataFrame({'Market_change': [1, 2, 3, 4, 5, 6, 7]})
        expected_changes = ["Down", "Down", "Down", "NC", "Up", "Up", "Up"]
        res = set_currency_change(df)
        self.assertEqual(expected_changes, res['Market_change'].values.tolist())

    def test_calculate_thresholds(self):
        df = pd.DataFrame({'Market_change': [1, 2, 3, 4, 5, 6, 7]})
        lower, higher = calculate_thresholds(df)
        self.assertEqual((3.33, 4.67), (lower, higher))

    def test_get_tweets_with_affect_filename(self):
        res = get_tweets_with_affect_filename("ABC")
        self.assertEqual(r"data_path_here\tweets_affect_ABC.csv", res)

    def test_get_graph_filename(self):
        res = get_graph_filename("ABC")
        self.assertEqual(r"data_path_here\graph_data_ABC.csv", res)

    def test_get_currency_prices_filename(self):
        res = get_currency_prices_filename("ABC")
        self.assertEqual(r"data_path_here\ABCIndex.csv", res)

    def test_get_graph_data(self):
        mock_data = pd.DataFrame({"Text": ["First", "Second"],
                                  "Date": ['2018-03-06 11:22:33', '2018-03-07 22:33:44'],
                                  "Open": [110, 120]})
        with mock.patch("pandas.read_csv", return_value=mock_data):
            dates, prices, tweets_per_date = get_graph_data("ABC")
            self.assertEqual([110, 120], prices)
            self.assertEqual(['2018-03-06 11:22:33', '2018-03-07 22:33:44'], dates)
            self.assertEqual({'2018-03-06 11:22:33': 'First', '2018-03-07 22:33:44': 'Second'}, tweets_per_date)

    # def test_save_sifted_tweets_with_date(self):
    #     input_df = pd.DataFrame({"Text": ["First", "Second"], "F1": [0, 0], "F2": [1, 1]})
    #     mock_curr_values = pd.DataFrame({"Date": ['Mar 06, 2018', 'Mar 07, 2018', 'Mar 08, 2018'], "Change": [1.3, 2.6, 0.7], "Price": 0, "Open": 0, "High": 0, "Low": 0, "Vol.": 0})
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

    def test_set_date_with_effect(self):
        df = pd.DataFrame({"Text": ["First", "Second"],
                           "Date": [pd.Timestamp('2017-01-02T23:47'), pd.Timestamp('2017-01-04T12:47')]})
        res = set_date_with_effect(df)
        expected_dates = [pd.Timestamp('2017-01-03 00:00:00'), pd.Timestamp('2017-01-04 00:00:00')]
        self.assertEqual(expected_dates, res["Date_with_affect"].tolist())
        self.assertEqual(["First", "Second"], res["Text"].tolist())


if __name__ == '__main__':
    unittest.main()
