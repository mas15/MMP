from datetime import timedelta
import pandas as pd
import os
from markets.sentiment import SentimentAnalyser
from markets.feature_extractor import FeatureExtractor

pd.set_option('display.width', 1500)

ALL_TWEETS_FILE = os.path.join(os.path.dirname(__file__), "data/all_tweets.csv")
USD_INDEX_FILE = os.path.join(os.path.dirname(__file__), "data/USDIndex.csv")
FEATURES_WITH_EFFECT_FILE = os.path.join(os.path.dirname(__file__), "data/features_with_effect.csv")
FEATURES_WITH_TEXT_AND_EFFECT_FILE = os.path.join(os.path.dirname(__file__), "data/text_with_feats_and_effect.csv")
MIN_FEATURE_OCCURENCIES = 7 # todo 6?


def read_all_tweets():
    all_tweets = pd.read_csv(ALL_TWEETS_FILE)
    all_tweets['Date'] = pd.to_datetime(all_tweets['Date'], format='%Y-%m-%d %H:%M:%S')
    return all_tweets


def read_dollar_prices():
    dollar_prices = pd.read_csv(USD_INDEX_FILE)
    dollar_prices['Date'] = pd.to_datetime(dollar_prices['Date'], format='%b %d, %Y')
    dollar_prices = dollar_prices[(dollar_prices['Date'].dt.year >= 2017)]
    # dollar_prices.set_index('Date', inplace=True)
    dollar_prices.drop(columns=['Vol.'], inplace=True)
    dollar_prices.rename(columns={'Change': 'Market_change'}, inplace=True)
    return dollar_prices


def get_date_to_check_affect(d):
    res = d if d.hour < 22 else d + timedelta(days=1)
    return res.normalize()


def set_date_with_effect(all_tweets, dollar_prices):
    all_tweets["Date_with_affect"] = all_tweets["Date"].apply(get_date_to_check_affect)
    all_tweets.sort_values(by='Date_with_affect', inplace=True)
    all_tweets.drop(columns=['Date', 'Id'], inplace=True)

    dollar_prices.sort_values(by='Date', inplace=True)  # nie sa takie same?

    result = pd.merge_asof(all_tweets, dollar_prices, left_on='Date_with_affect', right_on='Date', direction='forward')
    result.drop(columns=['Price', 'Open', 'High', 'Low', 'Date', 'Date_with_affect'], inplace=True)
    return result


def calculate_sentiment(tweets_df, sent):
    tweets_df["Tweet_sentiment"] = tweets_df["Text"].apply(sent.predict_score)
    #tweets_df["Tweet_sentiment"].replace({"pos": 1, "neg": 0}, inplace=True)
    return tweets_df


def create_feature_extractor(df):
    extr = FeatureExtractor(min_keyword_frequency=4)
    extr.build_vocabulary(df["Text"].tolist())
    return extr


def mark_row(row, extr):  # todo mark features on df
    features = extr.extract_features(row['Text'])
    for f, is_in_tweet in features.items():
        if is_in_tweet:
            row[f] = 1
    return row


def mark_features(extr, df):
    for f in extr.features:
        df[f] = 0

    df = df.apply(lambda row: mark_row(row, extr), axis=1)
    return df


def set_currency_change(result):
    def _get_change(x):
        if x > up_min:
            return "Up"
        elif x < down_max:
            return "Down"
        else:
            return "NC"

    down_max, up_min = calculate_thresholds(result)
    print(down_max)
    print(up_min)
    result["Market_change"] = result["Market_change"].apply(_get_change)
    return result


def move_column_to_the_end(df, col_name):
    cols = list(df)
    cols.append(cols.pop(cols.index(col_name)))
    df = df.reindex(columns=cols)
    return df


def count_nr_of_feature_occurencies(features): # todo test
    return [(col, (features.loc[features[col] == True, col].count())) for col in features]


def drop_infrequent_features(df, min_freq=MIN_FEATURE_OCCURENCIES):
    features = df.drop(columns=["Market_change", "Tweet_sentiment", "Text"], errors="ignore")
    cols_with_nr_of_trues = count_nr_of_feature_occurencies(features)
    cols_to_drop = [c[0] for c in cols_with_nr_of_trues if c[1] < min_freq]  # i c!=change
    df.drop(columns=cols_to_drop, axis=1, inplace=True)
    return df


def drop_instances_without_features(df):
    df = df[(df.drop(columns=["Market_change", "Tweet_sentiment", "Text"]).T != 0).any()]
    return df


def calculate_thresholds(df):
    mean = df["Market_change"].mean()
    sigma = df["Market_change"].std(ddof=0)
    lower_threshold = (mean - (sigma/3)).round(2)
    higher_threshold = (mean + (sigma/3)).round(2)
    return lower_threshold, higher_threshold


def filter_features(df, features_to_leave):
    cols_to_leave = features_to_leave + ["Tweet_sentiment", "Market_change", "Text"]
    cols_to_drop = [c for c in list(df) if c not in cols_to_leave]
    df.drop(columns=cols_to_drop, axis=1, inplace=True)
    return df


if __name__ == "__main__":
    all_tweets = read_all_tweets()
    dollar_prices = read_dollar_prices()
    sent = SentimentAnalyser()
    sent.load()

    result = set_date_with_effect(all_tweets, dollar_prices)
    # result.to_csv("data/tweets_with_prices.csv", index=False)

    result = set_currency_change(result)
    print("Dollar change set")

    extr = create_feature_extractor(result)
    result = mark_features(extr, result)
    print("Features marked")

    result = calculate_sentiment(result, sent)
    print("Sentiment calculated")

    # result = drop_infrequent_features(result)

    features_to_leave = [line.strip() for line in open("data/attr_po_6_wr_nb_bf_nc", 'r')]
    result = filter_features(result, features_to_leave)

    extr.set_features(features_to_leave)
    result = mark_features(extr, result)

    # TODO text jako index
    result = drop_instances_without_features(result)

    result = move_column_to_the_end(result, "Market_change")

    result.to_csv(FEATURES_WITH_TEXT_AND_EFFECT_FILE, index=False)

    result.drop(columns=['Text'], inplace=True) # todo set as index?
    result.to_csv(FEATURES_WITH_EFFECT_FILE, index=False)
