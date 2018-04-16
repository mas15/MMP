import os
import pandas as pd
from markets.phrases_extractor import PhrasesExtractor
from markets.sentiment import SentimentAnalyser
from markets.dataset import TweetsDataSet

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
ALL_TWEETS_FILENAME = os.path.join(DATA_PATH, "all_tweets.csv")
TWEETS_WITH_FEATURES_FILENAME = os.path.join(DATA_PATH, "tweets_with_features.csv")
MIN_FEATURE_OCCURENCIES = 7


def read_all_tweets(tweets_filename):
    all_tweets = pd.read_csv(tweets_filename)
    all_tweets['Date'] = pd.to_datetime(all_tweets['Date'], format='%Y-%m-%d %H:%M:%S')
    all_tweets.drop(columns=['Id'], inplace=True)
    return all_tweets


class FeatureExtractor:
    def __init__(self, df):
        self.df = df
        self.extr = PhrasesExtractor(min_keyword_frequency=4)  # 5 tez jest spoko
        self.extr.build_vocabulary(self.df.get_all_tweets())

        self.sent = SentimentAnalyser()
        self.sent.load()

    def extract_features(self):
        self.df.set_phrase_features(self.extr.extract_features)
        self.drop_infrequent_features()
        self.df.drop_instances_without_features()
        self.df.set_sentiment(self.sent.predict_score)
        return self.df

    def drop_infrequent_features(self):
        features = get_infrequent_features(self.df)
        self.df.remove_features(features)


def get_infrequent_features(dataset, min_freq=MIN_FEATURE_OCCURENCIES):
    cols_with_nr_of_trues = dataset.get_feature_occurencies()
    infrequent_features = [c[0] for c in cols_with_nr_of_trues if c[1] < min_freq]
    return infrequent_features


def build_tweets_features_dataframe():
    tweets_df = read_all_tweets(ALL_TWEETS_FILENAME)
    ds = TweetsDataSet(tweets_df)
    extractor = FeatureExtractor(ds)
    tweets_df_with_features = extractor.extract_features()
    tweets_df_with_features.save_to_csv(TWEETS_WITH_FEATURES_FILENAME)


if __name__ == '__main__':
    build_tweets_features_dataframe()
