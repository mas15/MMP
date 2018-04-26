import unittest
import pandas as pd
from unittest import mock
from parameterized import parameterized
from markets.sentiment import SentimentAnalyser, get_tweets_with_sentiment_from_file, calculate_sentiment
from nltk.classify.naivebayes import DictionaryProbDist


class TestSentimentAnalyser(unittest.TestCase):
    def setUp(self):
        mock_extractor = TestExtractor()
        mock_classifier = TestClassifier()
        self.sent = SentimentAnalyser(mock_extractor, mock_classifier)

    @parameterized.expand([
        ("Tweet content", "pos"),
        ("Negative content", "neg"),
    ])
    def test_predict(self, tweet, exp_result):
        res = self.sent.predict(tweet)
        self.assertEqual(exp_result, res)

    @parameterized.expand([
        ("Tweet content", 0.5),
        ("Negative content", 0.1),
    ])
    def test_predict_score(self, tweet, exp_result):
        res = self.sent.predict_score(tweet)
        self.assertEqual(exp_result, res)

    # def test_cross_validate_returns_average_accuracy(self):
    #     self.sent.train = mock.Mock(side_effect=[0.8, 0.9, 0.6, 0.3, 0.3])
    #     dataset = [(i, i) for i in range(15)]
    #     res = self.sent.cross_validate(dataset, 5)
    #     self.assertEqual(0.58, res)

    def test_get_tweets_with_sentiment_from_file(self):
        data = "1234,\"First tweet\",2016-12-30 19:41:33,pos\n1235,\"Second tweet\",2016-12-30 19:41:33,neg"
        m = mock.mock_open(read_data=data)
        m.return_value.__iter__ = lambda self: iter(self.readline, '')
        with mock.patch("builtins.open", m) as _:
            result = get_tweets_with_sentiment_from_file("any_filename")
            self.assertEqual([("First tweet", "pos"), ("Second tweet", "neg")], result)

    def test_get_tweets_with_sentiment_from_file_raises_when_wrong_sent(self):
        data = "1234,\"First tweet\",2016-12-30 19:41:33,pos\n1235,\"Second tweet\",2016-12-30 19:41:33,abc"
        m = mock.mock_open(read_data=data)
        m.return_value.__iter__ = lambda self: iter(self.readline, '')
        with self.assertRaises(Exception):
            with mock.patch("builtins.open", m) as _:
                get_tweets_with_sentiment_from_file("any_filename")

    def test_calculate_sentiment(self):
        df = pd.DataFrame({"Text": ["Tweet content", "Other"]})
        res = calculate_sentiment(df, self.sent)
        self.assertEqual([['Tweet content', 0.5], ['Other', 0.1]], res.values.tolist())


class TestExtractor:
    def extract_features(self, text):
        return text


class TestClassifier:
    def classify(self, features):
        return "pos" if features == "Tweet content" else "neg"

    def prob_classify(self, features):
        if features == "Tweet content":
            return DictionaryProbDist({"pos": 0.5, "neg": 0.1})
        return DictionaryProbDist({"pos": 0.1, "neg": 0.5})


if __name__ == '__main__':
    unittest.main()
