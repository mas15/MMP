import os
import pandas as pd
from markets.phrases_extractor import PhrasesExtractor
from markets.sentiment import SentimentAnalyser

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
ALL_TWEETS_FILENAME = os.path.join(DATA_PATH, "all_tweets.csv")
TWEETS_WITH_FEATURES_FILENAME = os.path.join(DATA_PATH, "tweets_with_features.csv")
MIN_FEATURE_OCCURENCIES = 7


def get_infrequent_features(df, min_freq=MIN_FEATURE_OCCURENCIES):
    df_with_features = df.drop(columns=["Text", "Date"])
    cols_with_nr_of_trues = count_nr_of_feature_occurrences(df_with_features)
    infrequent_features = [c[0] for c in cols_with_nr_of_trues if c[1] < min_freq]  # i c!=change
    return infrequent_features


def count_nr_of_feature_occurrences(df):
    return [(col, (df.loc[df[col] == True, col].count())) for col in df]


def add_features(df, features_to_add):
    for f in features_to_add:
        df[f] = 0
    return df


def mark_features(df, selecting_function):
    df = df.apply(lambda row: mark_row(row, selecting_function), axis=1)
    return df


def mark_row(row, selecting_function):
    features = selecting_function(row['Text']).items()
    for f, is_in_tweet in features:
        if is_in_tweet:
            row[f] = 1
    return row


def drop_infrequent_features(df):
    features_to_remove = get_infrequent_features(df)
    sifted_df = df.drop(columns=features_to_remove, axis=1)
    return sifted_df


def drop_instances_without_features(df):
    return df[(df.drop(columns=["Text"]).T != 0).any()]


def read_all_tweets(tweets_filename):
    all_tweets = pd.read_csv(tweets_filename)
    all_tweets['Date'] = pd.to_datetime(all_tweets['Date'], format='%Y-%m-%d %H:%M:%S')
    all_tweets.drop(columns=['Id'], inplace=True)
    return all_tweets


def set_sentiment(df, sentiment_calc_function):
    df["Tweet_sentiment"] = df["Text"].apply(sentiment_calc_function)
    return df


def extract_features(df): # todo pipeline
    extr = PhrasesExtractor(min_keyword_frequency=4) # 5 tez jest spoko
    extr.build_vocabulary(df["Text"].tolist())

    sent = SentimentAnalyser()
    sent.load()

    df = add_features(df, extr.features)
    df = mark_features(df, extr.extract_features)
    df = drop_infrequent_features(df)
    df = drop_instances_without_features(df)
    df = set_sentiment(df, sent.predict_score)
    return df


def build_tweets_features_dataframe():
    tweets_df = read_all_tweets(ALL_TWEETS_FILENAME)
    tweets_df_with_features = extract_features(tweets_df)
    tweets_df_with_features.to_csv(TWEETS_WITH_FEATURES_FILENAME, index=False)


if __name__ == '__main__':
    build_tweets_features_dataframe()
