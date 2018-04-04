import os
from datetime import timedelta
import pandas as pd
from markets.helpers import move_column_to_the_end

pd.set_option('display.width', 1500)

ALL_TWEETS_FILE = os.path.join(os.path.dirname(__file__), "data/all_tweets.csv")
USD_INDEX_FILE = os.path.join(os.path.dirname(__file__), "data/USDIndex.csv")
TWEETS_WITH_MARKET_CHANGE = os.path.join(os.path.dirname(__file__), "data/tweets_with_effect.csv")
GRAPH_DATA_FILE = os.path.join(os.path.dirname(__file__), "data/graph_data.csv")


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


def set_date_with_effect(all_tweets):
    all_tweets["Date_with_affect"] = all_tweets["Date"].apply(get_date_to_check_affect)
    all_tweets.sort_values(by='Date_with_affect', inplace=True)
    all_tweets.drop(columns=['Date', 'Id'], inplace=True)
    return all_tweets


def merge_tweets_with_dollar_prices(all_tweets, dollar_prices, drop_open_and_date=True):
    dollar_prices.sort_values(by='Date', inplace=True)  # nie sa takie same?

    result = pd.merge_asof(all_tweets, dollar_prices, left_on='Date_with_affect', right_on='Date', direction='forward')
    columns_to_drop = ['Price', 'High', 'Low', 'Date_with_affect']
    if drop_open_and_date:
        columns_to_drop += ['Open', 'Date']
    result.drop(columns=columns_to_drop, inplace=True)
    # todo powinno sprawdzac czy wszystko sie dopasowało
    return result


def set_currency_change(result):
    def _get_change(x):
        if x > up_min:
            return "Up"
        if x < down_max:
            return "Down"
        return "NC"

    down_max, up_min = calculate_thresholds(result)
    print(down_max)
    print(up_min)
    result["Market_change"] = result["Market_change"].apply(_get_change)
    return result


def calculate_thresholds(df):
    mean = df["Market_change"].mean()
    sigma = df["Market_change"].std(ddof=0)
    lower_threshold = (mean - (sigma / 3)).round(2)
    higher_threshold = (mean + (sigma / 3)).round(2)
    return lower_threshold, higher_threshold


def save_sifted_tweets_with_date(df):
    # tweety + date_with_effect -> open value
    all_tweets = read_all_tweets()
    dollar_prices = read_dollar_prices()
    tweets_with_date = set_date_with_effect(all_tweets)  # todo do jednej funckji - read_set
    result = merge_tweets_with_dollar_prices(tweets_with_date, dollar_prices, False)

    # dodac open value
    result_df = result[result["Text"].isin(df["Text"])]
    result_df["Date"] = result_df["Date"].dt.strftime('%Y-%m-%d')
    result_df.to_csv(GRAPH_DATA_FILE, index=False)
    return result_df


def get_graph_data():
    graph_data = pd.read_csv(GRAPH_DATA_FILE)
    tweets_per_date = dict(zip(graph_data.Date, graph_data.Text))
    dates = graph_data["Date"].values.tolist()
    prices = graph_data["Open"].values.tolist()

    return dates, prices, tweets_per_date


def build_df_with_tweets_and_effect():
    all_tweets = read_all_tweets()
    dollar_prices = read_dollar_prices()

    all_tweets = set_date_with_effect(all_tweets)
    result = merge_tweets_with_dollar_prices(all_tweets, dollar_prices)
    # result.to_csv("data/tweets_with_prices.csv", index=False)

    result = set_currency_change(result)
    print("Dollar change set")

    result = move_column_to_the_end(result, "Market_change")  # todo czy to potrzebne?

    result.to_csv(TWEETS_WITH_MARKET_CHANGE, index=False)

    # result.drop(columns=['Text'], inplace=True) # todo set as index?
    # result.to_csv(FEATURES_WITH_EFFECT_FILE, index=False)


if __name__ == "__main__":
    build_df_with_tweets_and_effect()
