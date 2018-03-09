from datetime import timedelta

import pandas as pd

from sentiment import SentimentAnalyser

pd.set_option('display.width', 1500)


def read_all_tweets():
    all_tweets = pd.read_csv("all_tweets.csv")
    all_tweets['Date'] = pd.to_datetime(all_tweets['Date'], format='%Y-%m-%d %H:%M:%S')
    # all_tweets.set_index('Id', inplace=True)   
    return all_tweets


def read_dollar_prices():
    dollar_prices = pd.read_csv("USDIndex.csv")
    dollar_prices['Date'] = pd.to_datetime(dollar_prices['Date'], format='%b %d, %Y')
    # dollar_prices.set_index('Date', inplace=True)
    dollar_prices.drop(columns=['Vol.'], inplace=True)
    return dollar_prices


def get_date_to_check_affect(d):
    res = d if d.hour < 22 else d + timedelta(days=1)
    return res.normalize()


all_tweets = read_all_tweets()
dollar_prices = read_dollar_prices()
sent = SentimentAnalyser()
sent.load()

all_tweets["Date_with_affect"] = all_tweets["Date"].apply(get_date_to_check_affect)
all_tweets.sort_values(by='Date_with_affect', inplace=True)
all_tweets.drop(columns=['Id', 'Date'], inplace=True)

dollar_prices.sort_values(by='Date', inplace=True)  # nie sa takie same?

tweets_with_affect = pd.merge_asof(all_tweets, dollar_prices, left_on='Date_with_affect', right_on='Date',
                                   direction='forward')
tweets_with_affect.drop(columns=['Price', 'Open', 'High', 'Low', 'Date', 'Date_with_affect'], inplace=True)


# tweets_with_affect["Sentiment"] = tweets_with_affect["Text"].apply(sent.analyse)
print(tweets_with_affect.head())

for feature in sent.extr.phrases + sent.extr.vocabulary:
    tweets_with_affect[feature] = False


def mark_features(row):
    features = sent.extr.extract_features(row['Text'])
    for f, is_in_tweet in features.items():
        if is_in_tweet:
            row[f] = True


tweets_with_affect.apply(mark_features, axis=1)
features_with_affect = tweets_with_affect.drop(columns=['Text'])

print("RESULT--------------------------------------")
print(features_with_affect.head())
print(features_with_affect.bill.sum())
print(features_with_affect.person.sum())
