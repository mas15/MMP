import unittest
from unittest import mock
from unittest.mock import create_autospec
from parameterized import parameterized
from markets.rules import group_into_sets
import pandas as pd


class TestRuleLearning(unittest.TestCase):

    def test_group_into_sets(self):
        r1 = dict({"words_set": {"a", "b", "c"}, "support": 11, "confidence": 11, "antecedants": ["a", "b"], "consequents": ["c"]})
        r2 = dict({"words_set": {"a", "b", "c"}, "support": 11, "confidence": 11, "antecedants": ["a", "c"], "consequents": ["b"]})
        r3 = dict({"words_set": {"b", "c", "a"}, "support": 11, "confidence": 11, "antecedants": ["c", "b"], "consequents": ["a"]})
        r4 = dict({"words_set": {"d", "e", "f"}, "support": 121, "confidence": 121, "antecedants": ["d", "e"], "consequents": ["f"]})
        rules = [r1, r2, r3, r4]
        result = group_into_sets(rules)
        self.assertEquals(2, len(result))
        self.assertEquals({"a", "b", "c"}, result[0].words_set)
        self.assertEquals({"d", "e", "f"}, result[1].words_set)