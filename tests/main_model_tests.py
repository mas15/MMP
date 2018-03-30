import unittest
from markets.main_model_build import put_results_in_dict, get_misclassified_on_set, get_indexes_before_splitting, \
    sort_misclassified
import numpy as np
import pandas as pd


class TestMainModel(unittest.TestCase):
    def test_get_misclassified_objects(self):
        y = np.array(["Up", "NC", "Down", "Up"])
        predicted = np.array(["NC", "NC", "Down", "Down"])
        result = get_misclassified_on_set(y, predicted)
        self.assertEqual([0, 3], result[0].tolist()) # mozoe niech zwaraca z [0]?

    def test_get_indexes_before_splitting(self):  # todo indexowane od 1?
        train_indexes = np.array([0, 1, 3, 5, 6, 7, 8])
        test_indexes = np.array([2, 4, 9])
        misclass_train = np.array([2, 3, 6])  # 3, 5, 8
        misclass_test = np.array([1])  # 4

        res = get_indexes_before_splitting(train_indexes, misclass_train)
        self.assertEqual([3, 5, 8], res.tolist())

        res = get_indexes_before_splitting(test_indexes, misclass_test)
        self.assertEqual([4], res.tolist())

    def test_put_results_in_dict(self):
        features = pd.DataFrame({'f1': [0], 'f2': [1], 'f3': [0], 'f4': [1], "Tweet_sentiment": [0.23]})
        propabs = [("Down", 0.1), ("NC", 0.5), ("UP", 0.2)]
        prediction = "NC"
        res = put_results_in_dict(prediction, propabs, features)
        exp_res = {'Down': 0.1, 'NC': 0.5, 'UP': 0.2, 'prediction': 'NC', 'features': ['f2', 'f4'],
                   'sentiment': 'Negative'}
        self.assertEqual(exp_res, res)

    def test_sort_misclassified(self):
        misclassified_objects = dict([(1, 12), (2, 0), (3, 343), (4, 1), (5, 100)])
        res = sort_misclassified(misclassified_objects)
        exp_res = [(3, 343), (5, 100), (1, 12), (4, 1)]
        self.assertEqual(exp_res, res)


if __name__ == '__main__':
    unittest.main()
