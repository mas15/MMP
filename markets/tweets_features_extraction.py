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


#
# class FeatureExtractor:
#     def __init__(self, extr=None, sent=None, min_freq=MIN_FEATURE_OCCURRENCES):
#         self.extr = extr or PhrasesExtractor(min_keyword_frequency=4)
#         self.sent = sent or SentimentAnalyser()
#         self.sent.load()
#         self.min_feature_freq = min_freq
#
#     def _set_vocabulary(self, dataset, vocabulary):
#         if vocabulary:
#             self.extr.set_features(vocabulary)
#         else:
#             self.extr.build_vocabulary(dataset.get_all_tweets())
#
#
#     def extract_features(self, dataset, vocabulary=None, with_dropping=True):
#         self._set_vocabulary(dataset, vocabulary)
#         dataset.set_phrase_features(self.extr.extract_features, self.extr.features)
#         if with_dropping:
#             self.drop_infrequent_features(dataset)
#             dataset.drop_instances_without_features()
#         dataset.set_sentiment(self.sent.predict_score)
#         return dataset

def remark_features(dataset, vocabulary=None, with_dropping=True):
    extr = PhrasesExtractor(min_keyword_frequency=4)
    if vocabulary:
        extr.set_features(vocabulary)
    else:
        extr.build_vocabulary(dataset.get_all_tweets())

    dataset.set_phrase_features(extr.extract_features, extr.features)
    if with_dropping:
        dataset.drop_instances_without_features()
    return dataset


def drop_infrequent_features(dataset, min_feature_freq):
    cols_with_nr_of_trues = dataset.get_feature_occurrences()
    infrequent_features = [c[0] for c in cols_with_nr_of_trues if c[1] < min_feature_freq]
    dataset.remove_features(infrequent_features)


def extract_features(dataset, vocabulary=None, with_dropping=True, min_freq=MIN_FEATURE_OCCURRENCES):
    extr = PhrasesExtractor(min_keyword_frequency=4)
    if vocabulary:
        extr.set_features(vocabulary)
    else:
        extr.build_vocabulary(dataset.get_all_tweets())

    sent = SentimentAnalyser()  # TODO load when made
    sent.load()

    dataset.set_phrase_features(extr.extract_features, extr.features)
    if with_dropping:
        drop_infrequent_features(dataset, min_freq)
        dataset.drop_instances_without_features()
    dataset.set_sentiment(sent.predict_score)
    return dataset


def build_tweets_features_dataframe():
    tweets_df = read_all_tweets(ALL_TWEETS_FILENAME)
    ds = TweetsDataSet(tweets_df)
    tweets_df_with_features = extract_features(ds)
    tweets_df_with_features.save_to_csv(TWEETS_WITH_FEATURES_FILENAME)


def build_dataset_with_one_tweet(text, features, min_freq=MIN_FEATURE_OCCURRENCES):
    tweet_dataset = TweetsDataSet(pd.DataFrame({'Text': [text]}))
    tweet_dataset_with_features = extract_features(tweet_dataset, features, False, min_freq)  # todo nad tym pomyśleć
    return tweet_dataset_with_features


if __name__ == '__main__':
    build_tweets_features_dataframe()
