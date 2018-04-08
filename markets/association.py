from datetime import timedelta
import pandas as pd
from markets.helpers import move_column_to_the_end

pd.set_option('display.width', 1500)


def read_all_tweets(tweets_filename):
    all_tweets = pd.read_csv(tweets_filename)
    all_tweets['Date'] = pd.to_datetime(all_tweets['Date'], format='%Y-%m-%d %H:%M:%S')
    all_tweets.drop(columns=['Id'], inplace=True)  # todo czy to dobrze?
    return all_tweets


def read_currency_prices(prices_filename):
    prices = pd.read_csv(prices_filename)

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


def save_sifted_tweets_with_date(df, tweets_filename, prices_filename, output_filename):  # todo czy to tu?
    result = get_tweets_with_currency_prices(tweets_filename, prices_filename, False)
    result_df = result[result["Text"].isin(df["Text"])].copy()
    result_df["Date"] = result_df["Date"].dt.strftime('%Y-%m-%d')
    result_df.to_csv(output_filename, index=False)  # todo test czy z dobra nazwa wywoalane
    return result_df


def build_df_with_tweets_and_effect(tweets_filename, prices_filename):  # todo test
    result = get_tweets_with_currency_prices(tweets_filename, prices_filename)
    result = set_currency_change(result)
    result = move_column_to_the_end(result, "Market_change")  # todo czy to potrzebne?
    return result


def get_tweets_with_currency_prices(tweets_filename, prices_filename, drop_open_and_date=True):  # todo test?
    all_tweets = read_all_tweets(tweets_filename)
    currency_prices = read_currency_prices(prices_filename)
    tweets_with_date = set_date_with_effect(all_tweets)
    result = merge_tweets_with_dollar_prices(tweets_with_date, currency_prices, drop_open_and_date)
    return result
