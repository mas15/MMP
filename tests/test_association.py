from markets.association import get_date_to_check_affect, set_currency_change, calculate_thresholds
import pandas as pd
import unittest


class TestAssosiationLearning(unittest.TestCase): #todo rename
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

    # todo set_date_with_effect, read_all_tweets, read_dollar_prices,


if __name__ == '__main__':
    unittest.main()
