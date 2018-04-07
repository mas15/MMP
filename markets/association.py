import os
from datetime import timedelta
import pandas as pd
from markets.helpers import move_column_to_the_end, DATA_PATH

pd.set_option('display.width', 1500)

GRAPH_DATA_FILE_PREFIX = "graph_data_"
AFFECT_FILE_PREFIX = "tweets_affect_"
CURRENCY_PRICES_FILE_SUFFIX = "Index.csv"
ALL_TWEETS_FILE = os.path.join(DATA_PATH, "all_tweets.csv")


def read_all_tweets():
    all_tweets = pd.read_csv(ALL_TWEETS_FILE)
    all_tweets['Date'] = pd.to_datetime(all_tweets['Date'], format='%Y-%m-%d %H:%M:%S')
    all_tweets.drop(columns=['Id'], inplace=True) # todo czy to dobrze?
    return all_tweets


def read_currency_prices(currency):
    filename = get_currency_prices_filename(currency)
    prices = pd.read_csv(filename)

    prices['Date'] = pd.to_datetime(prices['Date'], format='%b %d, %Y')
    prices = prices[(prices['Date'].dt.year >= 2017)]
    # dollar_prices.set_index('Date', inplace=True)

    result = prices.filter(['Text', 'Date', 'Open', 'Change'], axis=1)
    result.rename(columns={'Change': 'Market_change'}, inplace=True)
    return result


def get_date_to_check_affect(d):
    res = d if d.hour < 22 else d + timedelta(days=1)
    return res.normalize()


def set_date_with_effect(tweets_df):
    tweets_df["Date_with_affect"] = tweets_df["Date"].apply(get_date_to_check_affect)
    tweets_df.sort_values(by='Date_with_affect', inplace=True)
    tweets_df.drop(columns=['Date'], inplace=True)
    return tweets_df


def merge_tweets_with_dollar_prices(all_tweets, dollar_prices, drop_open_and_date=True):
    dollar_prices.sort_values(by='Date', inplace=True)  # nie sa takie same?

    result = pd.merge_asof(all_tweets, dollar_prices, left_on='Date_with_affect', right_on='Date', direction='forward')

    columns_to_drop = ['Date_with_affect', 'Open', 'Date'] if drop_open_and_date else ['Date_with_affect']
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


def save_sifted_tweets_with_date(df, currency):
    all_tweets = read_all_tweets()
    dollar_prices = read_currency_prices(currency)
    tweets_with_date = set_date_with_effect(all_tweets)  # todo do jednej funckji - read_set

    result = merge_tweets_with_dollar_prices(tweets_with_date, dollar_prices, False)

    result_df = result[result["Text"].isin(df["Text"])].copy()
    result_df["Date"] = result_df["Date"].dt.strftime('%Y-%m-%d')
    result_df.to_csv(get_graph_filename(currency), index=False)  # todo test czy z dobra nazwa wywoalane
    return result_df


def get_tweets_with_affect_filename(currency):
    return os.path.join(DATA_PATH, AFFECT_FILE_PREFIX + currency + ".csv")


def get_graph_filename(currency):
    return os.path.join(DATA_PATH, GRAPH_DATA_FILE_PREFIX + currency + ".csv")


def get_currency_prices_filename(currency):
    return os.path.join(DATA_PATH, currency + CURRENCY_PRICES_FILE_SUFFIX)


def get_graph_data(currency):
    graph_data = pd.read_csv(get_graph_filename(currency))
    tweets_per_date = dict(zip(graph_data.Date, graph_data.Text))
    dates = graph_data["Date"].values.tolist()
    prices = graph_data["Open"].values.tolist()

    return dates, prices, tweets_per_date


def build_df_with_tweets_and_effect(currency): # todo test
    all_tweets = read_all_tweets()
    prices = read_currency_prices(currency)

    all_tweets = set_date_with_effect(all_tweets)
    result = merge_tweets_with_dollar_prices(all_tweets, prices)
    # result.to_csv("data/tweets_with_prices.csv", index=False)

    result = set_currency_change(result)
    print("Currency change set")

    result = move_column_to_the_end(result, "Market_change")  # todo czy to potrzebne?

    result_filename = get_tweets_with_affect_filename(currency)
    result.to_csv(result_filename, index=False)

    # result.drop(columns=['Text'], inplace=True) # todo set as index?
    # result.to_csv(FEATURES_WITH_EFFECT_FILE, index=False)
    return result_filename


# if __name__ == "__main__":
#    build_df_with_tweets_and_effect()
