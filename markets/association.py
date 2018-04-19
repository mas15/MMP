from datetime import timedelta
import pandas as pd

from markets.dataset import TweetsDataSet

pd.set_option('display.width', 1500)


def move_column_to_the_end(df, col_name):
    cols = list(df)
    cols.append(cols.pop(cols.index(col_name)))
    df = df.reindex(columns=cols)
    return df


def read_tweets_with_features(tweets_filename): # todo https://stackoverflow.com/questions/17465045/can-pandas-automatically-recognize-dates
    all_tweets = pd.read_csv(tweets_filename)
    all_tweets['Date'] = pd.to_datetime(all_tweets['Date'], format='%Y-%m-%d %H:%M:%S')
    return all_tweets


def read_currency_prices(prices_filename):
    prices = pd.read_csv(prices_filename)

    prices['Date'] = pd.to_datetime(prices['Date'], format='%b %d, %Y')
    prices = prices[(prices['Date'].dt.year >= 2017)]
    # dollar_prices.set_index('Date', inplace=True)

    result = prices.filter(['Text', 'Date', 'Open', 'Change'], axis=1)
    result.rename(columns={'Change': 'Market_change'}, inplace=True)
    return result


def calculate_thresholds(stock_prices):
    mean = stock_prices.mean()
    sigma = stock_prices.std(ddof=0)
    lower_threshold = (mean - (sigma / 3)).round(2)
    higher_threshold = (mean + (sigma / 3)).round(2)
    return lower_threshold, higher_threshold


def set_currency_change(dataset):
    def _get_change(x):
        if x > up_min:
            return "Up"
        if x < down_max:
            return "Down"
        return "NC"

    down_max, up_min = calculate_thresholds(dataset.get_market_change()) # todo to mozna do klasy wziac
    print(down_max)
    print(up_min)
    dataset.set_market_change(_get_change)


def save_sifted_tweets_with_date(sifted, tweets_filename, prices_filename, output_filename):
    result = get_tweets_with_currency_prices(tweets_filename, prices_filename, False)
    result.filter_by_tweets(sifted.get_all_tweets())
    result.df["Date"] = result.df["Date"].dt.strftime('%Y-%m-%d')
    result.df.to_csv(output_filename, index=False)  # todo test czy z dobra nazwa wywoalane
    return result.df


def get_tweets_with_currency_prices(tweets_filename, prices_filename, drop_open_and_date=True):  # todo test?
    all_tweets = read_tweets_with_features(tweets_filename)
    currency_prices = read_currency_prices(prices_filename)

    ds = TweetsDataSet(all_tweets)
    ds.set_date_with_effect()
    ds.merge_tweets_with_dollar_prices(currency_prices, drop_open_and_date) # todo to jest średnie
    return ds


def build_df_with_tweets_and_affect(tweets_filename, prices_filename):  # todo test
    result = get_tweets_with_currency_prices(tweets_filename, prices_filename)
    set_currency_change(result)
    result.df = move_column_to_the_end(result.df, "Market_change")  # todo usunac
    return result
