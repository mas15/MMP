from markets.association import get_date_to_check_affect, mark_features
import pandas as pd
import unittest


class TestAssosiationLearning(unittest.TestCase):
    def test_get_date_to_check_affect(self):
        d = pd.Timestamp('2017-01-02T23:47')
        exp_res = pd.Timestamp('2017-01-03')
        res = get_date_to_check_affect(d)
        self.assertEqual(exp_res, res)

    class MockFeatExtr:
        def extract_features(self, text):
            res = {"a": {"One": 1}, "b": {"Two": 1, "One": 1}, "c": {}} # co jak np "Another": 1
            return res[text]

    def test_mark_features(self):
        extr = self.MockFeatExtr()
        df = pd.DataFrame({"Text": ['a', 'b', 'c'], "One": [0, 0, 0], "Two": [0, 0, 0], "Three": [0, 0, 0]})
        result = df.apply(lambda x: mark_features(x, extr), axis=1)
        exp_df = pd.DataFrame({"Text": ['a', 'b', 'c'], "One": [1, 1, 0], "Two": [0, 1, 0], "Three": [0, 0, 0]})
        self.assertTrue(exp_df.equals(result))




if __name__ == '__main__':
    unittest.main()