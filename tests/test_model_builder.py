import unittest
from unittest import mock
from unittest.mock import create_autospec
from parameterized import parameterized
from markets.predicting_model_builder import ModelTrainer
from markets.main_model import AssociationDataProcessor,
import pandas as pd


class TestModelTrainer(unittest.TestCase):

    def setUp(self):
        mock_df_processor = create_autospec(AssociationDataProcessor)
        self.trainer = ModelTrainer(mock_df_processor)

    def test_train(self):
        df = pd.DataFrame({"Text": ["First", "Second"], "Market_change": [0.2, 0.5]})
        # todo
        #self.assertEqual(expected_result, result_df.to_dict())


    def test_get_features_iterator(self):
        #todo