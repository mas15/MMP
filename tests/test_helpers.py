import unittest
from unittest import mock
from unittest.mock import create_autospec
from markets.phrases_extractor import PhrasesExtractor
from markets.helpers import get_x_y_from_df, move_column_to_the_end, get_x_y_from_list_of_tuples
import pandas as pd


class TestHelpers(unittest.TestCase):
    def test_get_x_y_from_list_of_tuples(self):
        x, y = [1, 2, 3], [1, 4, 9]
        res_x, res_y = get_x_y_from_list_of_tuples(zip(x, y))
        self.assertEqual((x, y), (res_x.tolist(), res_y.tolist()))

    def test_move_column_to_the_end(self):
        df = pd.DataFrame(columns=["A", "B", "C", "D"])
        res = move_column_to_the_end(df, "B")
        self.assertEqual(["A", "C", "D", "B"], res.columns.tolist())
