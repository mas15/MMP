from datetime import timedelta
import pandas as pd
import os
from markets.sentiment import SentimentAnalyser
from markets.feature_extractor import FeatureExtractor

pd.set_option('display.width', 1500)

ALL_TWEETS_FILE = os.path.join(os.path.dirname(__file__), "data/all_tweets.csv")
USD_INDEX_FILE = os.path.join(os.path.dirname(__file__), "data/USDIndex.csv")
FEATURES_WITH_EFFECT_FILE = os.path.join(os.path.dirname(__file__), "data/features_with_effect.csv")
MIN_FEATURE_OCCURENCIES = 6
CHANGE_TRESCHOLD = 0.1


def read_all_tweets():
    all_tweets = pd.read_csv(ALL_TWEETS_FILE)
    all_tweets['Date'] = pd.to_datetime(all_tweets['Date'], format='%Y-%m-%d %H:%M:%S')
    # all_tweets.set_index('Id', inplace=True)   
    return all_tweets


def read_dollar_prices():
    dollar_prices = pd.read_csv(USD_INDEX_FILE)
    dollar_prices['Date'] = pd.to_datetime(dollar_prices['Date'], format='%b %d, %Y')
    dollar_prices = dollar_prices[(dollar_prices['Date'].dt.year >= 2017)]
    # dollar_prices.set_index('Date', inplace=True)
    dollar_prices.drop(columns=['Vol.'], inplace=True)
    return dollar_prices


def get_date_to_check_affect(d):
    res = d if d.hour < 22 else d + timedelta(days=1)
    return res.normalize()


def mark_features(row, extr):
    features = extr.extract_features(row['Text'])
    for f, is_in_tweet in features.items():
        if is_in_tweet:
            row[f] = 1
    return row


def set_date_with_effect(all_tweets, dollar_prices):
    all_tweets["Date_with_affect"] = all_tweets["Date"].apply(get_date_to_check_affect)
    all_tweets.sort_values(by='Date_with_affect', inplace=True)
    all_tweets.drop(columns=['Id', 'Date'], inplace=True)

    dollar_prices.sort_values(by='Date', inplace=True)  # nie sa takie same?

    result = pd.merge_asof(all_tweets, dollar_prices, left_on='Date_with_affect', right_on='Date', direction='forward')
    result.drop(columns=['Price', 'Open', 'High', 'Low', 'Date', 'Date_with_affect'], inplace=True)
    return result


def calculate_sentiment(tweets_df):
    tweets_df["Sentiment"] = tweets_df["Text"].apply(sent.analyse)
    tweets_df["Sentiment"] = tweets_df["Sentiment"].replace({"pos": 1, "neg": 0})
    return tweets_df


def extract_features_from_text(result):
    extr = FeatureExtractor(min_keyword_frequency=4)
    extr.build_vocabulary(result["Text"].tolist())
    for f in extr.features:
        result[f] = 0

    # move mark which features are in a tweet
    result = result.apply(lambda x: mark_features(x, extr), axis=1)
    result.drop(columns=['Text'], inplace=True)
    return result


def set_currency_change(result):
    def get_change(x):
        if x > CHANGE_TRESCHOLD:
            return "Up"
        elif x < -CHANGE_TRESCHOLD:
            return "Down"
        else:
            return "NC"

    result["Change"] = result["Change"].apply(get_change)
    return result


def drop_infrequent_features(result):
    features = result.drop(columns=["Change", "Sentiment"])
    cols_with_nr_of_trues = [(col, (features.loc[features[col] == True, col].count())) for col in features]
    #cols_with_nr_of_trues.sort(key=lambda t: t[1])  # todo to mozna usunac
    cols_to_drop = [c[0] for c in cols_with_nr_of_trues if c[1] <= MIN_FEATURE_OCCURENCIES]  # i c!=change
    print("Dropping " + str(len(cols_to_drop)))
    print(cols_to_drop)
    result.drop(columns=cols_to_drop, axis=1, inplace=True)
    return result


if __name__ == "__main__":
    all_tweets = read_all_tweets()
    dollar_prices = read_dollar_prices()
    sent = SentimentAnalyser()
    sent.load()

    result = set_date_with_effect(all_tweets, dollar_prices)
    #result.to_csv("data/tweets_with_prices.csv", index=False)

    result = calculate_sentiment(result)
    print("Sentiment calculated")

    result = extract_features_from_text(result)
    print("Features marked")

    result = set_currency_change(result)
    print("Dollar change set")
    print(result.head())

    result = drop_infrequent_features(result)

    cols_to_leave = [line.strip() for line in open("data/attr_after_6_wr_nb_bf", 'r')]
    cols_to_leave += ["Sentiment", "Change"]
    cols_to_drop = [c for c in list(result) if c not in cols_to_leave]
    result.drop(columns=cols_to_drop, axis=1, inplace=True)

    # save to file
    result.to_csv(FEATURES_WITH_EFFECT_FILE, index=False)
