import os
import pandas as pd
from markets.phrases_extractor import PhrasesExtractor
from markets.sentiment import SentimentAnalyser
from markets.dataset import TweetsDataSet

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
ALL_TWEETS_FILENAME = os.path.join(DATA_PATH, "all_tweets.csv")
TWEETS_WITH_FEATURES_FILENAME = os.path.join(DATA_PATH, "tweets_with_features.csv")
MIN_FEATURE_OCCURRENCES = 7


def read_all_tweets(tweets_filename):
    all_tweets = pd.read_csv(tweets_filename)
    all_tweets['Date'] = pd.to_datetime(all_tweets['Date'], format='%Y-%m-%d %H:%M:%S')
    all_tweets.drop(columns=['Id'], inplace=True)
    return all_tweets


class FeatureExtractor:
    def __init__(self, dataset, vocabulary=None, extr=None, sent=None, min_freq=MIN_FEATURE_OCCURRENCES):
        self.dataset = dataset
        self.extr = extr or PhrasesExtractor(min_keyword_frequency=4)
        if vocabulary:
            self.extr.set_features(vocabulary)
        else:
            self.extr.build_vocabulary(self.dataset.get_all_tweets())

        self.sent = sent or SentimentAnalyser()
        self.sent.load()
        self.min_feature_freq = min_freq

    def remark_features(self, with_dropping=True):
        self.dataset.set_phrase_features(self.extr.extract_features, self.extr.features)
        if with_dropping:
            self.dataset.drop_instances_without_features()
        return self.dataset

    def extract_features(self, with_dropping=True):
        self.dataset.set_phrase_features(self.extr.extract_features, self.extr.features)
        if with_dropping:
            self.drop_infrequent_features()
            self.dataset.drop_instances_without_features()
        self.dataset.set_sentiment(self.sent.predict_score)
        return self.dataset

    def drop_infrequent_features(self):
        cols_with_nr_of_trues = self.dataset.get_feature_occurrences()
        infrequent_features = [c[0] for c in cols_with_nr_of_trues if c[1] < self.min_feature_freq]
        self.dataset.remove_features(infrequent_features)


def build_tweets_features_dataframe():
    tweets_df = read_all_tweets(ALL_TWEETS_FILENAME)
    ds = TweetsDataSet(tweets_df)
    extractor = FeatureExtractor(ds)
    tweets_df_with_features = extractor.extract_features()
    tweets_df_with_features.save_to_csv(TWEETS_WITH_FEATURES_FILENAME)


def build_dataset_with_one_tweet(text, features):
    tweet_dataset = TweetsDataSet(pd.DataFrame({'Text': [text]}))
    extractor = FeatureExtractor(tweet_dataset, features)
    tweet_dataset_with_features = extractor.extract_features(with_dropping=False) # todo nad tym pomyśleć
    return tweet_dataset_with_features


if __name__ == '__main__':
    build_tweets_features_dataframe()
