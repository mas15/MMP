import pandas as pd
import unittest
from unittest import mock
from unittest.mock import patch
from parameterized import parameterized
from markets.currency_analyser import CurrencyAnalyser, get_graph_filename, \
    get_currency_prices_filename, get_rules_filename


@patch("markets.currency_analyser.DATA_PATH", "data_path_here")
class TestCurrencyAnalyser(unittest.TestCase):
    def setUp(self):
        self.analyser = CurrencyAnalyser("ABC")

    def test_get_graph_data(self):
        mock_data = pd.DataFrame({"Text": ["First", "Second"],
                                  "Date": ['2018-03-06 11:22:33', '2018-03-07 22:33:44'],
                                  "Open": [110, 120]})
        with mock.patch("pandas.read_csv", return_value=mock_data):
            dates, prices, tweets_per_date = self.analyser.get_graph_data()
            self.assertEqual([110, 120], prices)
            self.assertEqual(['2018-03-06 11:22:33', '2018-03-07 22:33:44'], dates)
            self.assertEqual({'2018-03-06 11:22:33': 'First', '2018-03-07 22:33:44': 'Second'}, tweets_per_date)

    def test_get_rules_data(self):  # todo przeniesc
        mock_data = pd.DataFrame({"antecedants": ["travel ban, politically", "milk, bread"],
                                  "consequents": ["correct", "butter"],
                                  "support": [0.0047694753577106515, 0.00123456789],
                                  "lift": [78.625, 'inf']})
        with mock.patch("pandas.read_csv", return_value=mock_data):
            rules = self.analyser.get_rules_data()
            self.assertEqual(sorted(['antecedants', 'consequents', 'support', 'lift']), sorted(rules["columns"]))


@patch("markets.currency_analyser.DATA_PATH", "data_path_here")
class TestFilenameFunctions(unittest.TestCase):
    def test_get_tweets_with_affect_filename(self):
        res = get_rules_filename("ABC")
        self.assertEqual(r"data_path_here\ABC_rules.csv", res)

    def test_get_graph_filename(self):
        res = get_graph_filename("ABC")
        self.assertEqual(r"data_path_here\ABC_graph_data.csv", res)

    def test_get_currency_prices_filename(self):
        res = get_currency_prices_filename("ABC")
        self.assertEqual(r"data_path_here\ABCIndex.csv", res)

    def test_get_rules_filename(self):
        res = get_rules_filename("ABC")
        self.assertEqual(r"data_path_here\ABC_rules.csv", res)
