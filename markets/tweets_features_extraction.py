import os
from markets.phrases_extraction import PhrasesExtractor
from markets.sentiment import SentimentAnalyser
from markets.dataset import TweetsDataSet, dataset_from_text
from markets.utils import read_all_tweets

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
ALL_TWEETS_FILENAME = os.path.join(DATA_PATH, "all_tweets.csv")
TWEETS_WITH_FEATURES_FILENAME = os.path.join(DATA_PATH, "tweets_with_features.csv")
MIN_FEATURE_OCCURRENCES = 7


def _create_phrases_extractor(vocabulary, dataset):
    extr = PhrasesExtractor(min_keyword_frequency=4)
    if vocabulary:
        extr.set_features(vocabulary)
    else:
        extr.build(dataset.get_all_tweets())
    return extr


def remark_features(dataset, vocabulary=None, with_dropping=True):
    extr = _create_phrases_extractor(vocabulary, dataset)
    dataset.set_phrase_features(extr.extract_features, extr.features)
    if with_dropping:
        dataset.drop_instances_without_features()
    return dataset


def drop_infrequent_features(dataset, min_feature_freq):
    cols_with_nr_of_trues = dataset.get_feature_occurrences()
    infrequent_features = [c[0] for c in cols_with_nr_of_trues if c[1] < min_feature_freq]
    dataset.remove_features(infrequent_features)


def extract_features(dataset, vocabulary=None, with_dropping=True, min_freq=MIN_FEATURE_OCCURRENCES):
    extr = _create_phrases_extractor(vocabulary, dataset)
    sent = SentimentAnalyser()
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


def build_dataset_with_one_tweet(text, features):
    tweet_dataset = dataset_from_text(text)
    tweet_dataset_with_features = extract_features(tweet_dataset, features, False)
    return tweet_dataset_with_features


if __name__ == '__main__':
    build_tweets_features_dataframe()
