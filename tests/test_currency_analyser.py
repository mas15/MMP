import pandas as pd
import unittest
from unittest import mock
from unittest.mock import patch
from parameterized import parameterized
from markets.currency_analysis import CurrencyAnalyser
import markets


@patch("markets.currency_analysis.DATA_PATH", "data_path_here")
class TestCurrencyAnalyser(unittest.TestCase):
    def setUp(self):
        self.analyser = CurrencyAnalyser("ABC")
    #
    # def test_constructor(self):
    #     self.assertEqual(r"data_path_here\ABC_rules.csv", self.analyser.rules_filename)
    #     self.assertEqual(r"data_path_here\ABC_graph_data.csv", self.analyser.graph_filename)
    #     self.assertEqual(r"data_path_here\ABCIndex.csv", self.analyser.currency_prices_filename)
    #     self.assertEqual(r"data_path_here\ABC_rules.csv", self.analyser.rules_filename)
    #     self.assertEqual(r"data_path_here\ABC_rules.csv", self.analyser.model_filename)

    def test_get_graph_data(self):
        mock_data = pd.DataFrame({"Text": ["First", "Second", "Third"],
                                  "Date": ['2018-03-06 11:22:33', '2018-03-07 22:33:44', '2018-03-07 22:33:44'],
                                  "Open": [110, 120, 111]})
        with mock.patch("pandas.read_csv", return_value=mock_data):
            dates, prices, tweets_per_date = self.analyser.get_graph_data()
            self.assertEqual([110, 120, 111], prices)
            self.assertEqual(['2018-03-06 11:22:33', '2018-03-07 22:33:44', '2018-03-07 22:33:44'], dates)
            self.assertEqual(["First"], tweets_per_date['2018-03-06 11:22:33'])
            self.assertEqual(["Second", "Third"], tweets_per_date['2018-03-07 22:33:44'])

    def test_get_most_coefficient_features_raises_if_no_model_yet(self):
        with self.assertRaises(Exception):
            self._model.get_most_coefficient_features()

    def test_analyse_tweet_raises_if_no_model_yet(self):
        with self.assertRaises(Exception):
            self._model.analyse_tweet("aaa")
