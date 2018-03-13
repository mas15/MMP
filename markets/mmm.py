from datetime import timedelta
import pandas as pd
import os
from markets.sentiment import SentimentAnalyser

pd.set_option('display.width', 1500)

ALL_TWEETS_FILE = os.path.join(os.path.dirname(__file__), "data/all_tweets.csv")
USD_INDEX_FILE = os.path.join(os.path.dirname(__file__), "data/USDIndex.csv")
FEATURES_WITH_EFFECT_FILE = os.path.join(os.path.dirname(__file__), "data/features_with_effect.csv")


def read_all_tweets():
    all_tweets = pd.read_csv(ALL_TWEETS_FILE)
    all_tweets['Date'] = pd.to_datetime(all_tweets['Date'], format='%Y-%m-%d %H:%M:%S')
    # all_tweets.set_index('Id', inplace=True)   
    return all_tweets


def read_dollar_prices():
    dollar_prices = pd.read_csv(USD_INDEX_FILE)
    dollar_prices['Date'] = pd.to_datetime(dollar_prices['Date'], format='%b %d, %Y')
    # dollar_prices.set_index('Date', inplace=True)
    dollar_prices.drop(columns=['Vol.'], inplace=True)
    return dollar_prices


def get_date_to_check_affect(d):
    res = d if d.hour < 22 else d + timedelta(days=1)
    return res.normalize()


def mark_features(row):
    features = sent.extr.extract_features(row['Text'])
    for f, is_in_tweet in features.items():
        if is_in_tweet:
            row[f] = True
    return row


if __name__ == "__main__":
    all_tweets = read_all_tweets()
    dollar_prices = read_dollar_prices()
    sent = SentimentAnalyser()
    sent.load()

    all_tweets["Date_with_affect"] = all_tweets["Date"].apply(get_date_to_check_affect)
    all_tweets.sort_values(by='Date_with_affect', inplace=True)
    all_tweets.drop(columns=['Id', 'Date'], inplace=True)

    dollar_prices.sort_values(by='Date', inplace=True)  # nie sa takie same?
    # Text, Date with affect

    result = pd.merge_asof(all_tweets, dollar_prices, left_on='Date_with_affect', right_on='Date', direction='forward')

    result.to_csv("tweets_with_prices.csv", index=False)
    result.drop(columns=['Price', 'Open', 'High', 'Low', 'Date', 'Date_with_affect'], inplace=True)

    result["Sentiment"] = result["Text"].apply(sent.analyse)

    features = list(sent.extr.phrases) + list(sent.extr.vocabulary)
    for f in features:
        result[f] = False

    # move mark which features are in a tweet
    result = result.apply(lambda x: mark_features(x), axis=1)

    # Change the change into True/False (dollar up, down)
    result["Dollar_up"] = result["Change"].apply(lambda x: x > 0)
    result.drop(columns=['Text', 'Change'], inplace=True)
    print(result.head())

    result.to_csv(FEATURES_WITH_EFFECT_FILE, index=False)
