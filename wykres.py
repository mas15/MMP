from mmm import read_all_tweets, read_dollar_prices,get_date_to_check_affect
import numpy as np

all_tweets = read_all_tweets()
dollar_prices = read_dollar_prices()
all_tweets.drop(columns=["Id"], inplace=True)
dollar_prices.drop(columns=["Price", "High", "Low", "Change"], inplace=True)

all_tweets["Date"] = all_tweets["Date"].apply(get_date_to_check_affect)

print(dict(zip(all_tweets.Date, all_tweets.Text)))

d = [x.isoformat() for x in dollar_prices["Date"].values]
v = [x for x in dollar_prices["Open"].values]
print(d)
print(v)
