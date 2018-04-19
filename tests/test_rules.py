import unittest
from markets.rules import group_into_sets, process_df
import pandas as pd


class TestRuleLearning(unittest.TestCase):

    def test_group_into_sets(self):
        r1 = dict({"words_set": {"a", "b", "c"}, "support": 11, "antecedants": ["a", "b"], "consequents": ["c"]})
        r2 = dict({"words_set": {"a", "b", "c"}, "support": 11, "antecedants": ["a", "c"], "consequents": ["b"]})
        r3 = dict({"words_set": {"b", "c", "a"}, "support": 11, "antecedants": ["c", "b"], "consequents": ["a"]})
        r4 = dict({"words_set": {"d", "e", "f"}, "support": 121, "antecedants": ["d", "e"], "consequents": ["f"]})
        rules = [r1, r2, r3, r4]
        for r in rules:
            r["lift"], r["confidence"], r["antecedent support"], r["consequent support"] = 11, 33, 0.66, 0.77
        result = group_into_sets(rules)
        self.assertEquals(2, len(result))
        self.assertEquals({"a", "b", "c"}, result[0].words_set)
        self.assertEquals({"d", "e", "f"}, result[1].words_set)

    def test_process_df(self):
        df = pd.DataFrame({"support": 11, "antecedants": [["a", "b"]],
                           "consequents": [["c"]], "lift": 0.123456789, "leverage": 1, "conviction": 1})
        expected_result = {'antecedants': 'a, b', 'consequents': 'c', 'lift': 0.12345679,
                           'support': 11, 'words_set': {'a', 'b', 'c'}}
        result = process_df(df)
        self.assertEquals(expected_result, result.iloc[0].to_dict())