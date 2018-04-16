import pandas as pd
from datetime import timedelta


class TweetsDataSet:
    non_text_feature_columns = ["Text", "Market_change", "Tweet_sentiment", "Date"]

    def __init__(self, df):
        self.df = df

    @property
    def features(self):
        return list(self.get_features_df())

    def get_features_df(self):
        return self.df.drop(columns=self.non_text_feature_columns, errors='ignore', axis=1)

    def get_no_features_df(self):
        return self.df[["Text", "Tweet_sentiment", "Market_change"]]

    def save_to_csv(self, filename):
        self.df.to_csv(filename, index=False)

    def set_phrase_features(self, selecting_function):
        self.df = self.df.apply(lambda row: self._mark_row(row, selecting_function), axis=1)
        self.df.fillna(0, inplace=True)

    def get_all_tweets(self):
        all_tweets = self.df["Text"].tolist()
        if not all_tweets:
            raise Exception("There is no tweets in the dataset")
        return all_tweets

    def set_sentiment(self, sentiment_calc_function):
        self.df["Tweet_sentiment"] = self.df["Text"].apply(sentiment_calc_function)

    @staticmethod
    def _mark_row(row, selecting_function):
        features = selecting_function(row['Text']).items()
        for f, is_in_tweet in features:
            if is_in_tweet:
                row[f] = 1
        return row

    def get_x_y(self):
        y = self.df["Market_change"].values.ravel()
        x = self.df.drop(columns=["Market_change", "Text"]).values  # todo text jakos ogarnac
        return x, y

    def get_feature_occurencies(self):
        features_with_occurencies = count_nr_of_feature_occurrences(self.get_features_df())
        return features_with_occurencies

    def _check_if_features_are_in_dataframe(self, features):
        feats_not_in_df = [f for f in features if f not in self.features]
        if feats_not_in_df:
            raise Exception("There are {0} selected features that are not in the dataset: {1}".format(len(feats_not_in_df), feats_not_in_df))

    def remove_features(self, features_to_remove):
        self._check_if_features_are_in_dataframe(features_to_remove)
        self.df.drop(columns=features_to_remove, axis=1, inplace=True)

    def drop_instances_without_features(self):
        self.df = self.df[(self.get_features_df().T != 0).any()]

    def set_date_with_effect(self):
        self.df["Date_with_affect"] = self.df["Date"].apply(get_date_to_check_affect)
        self.df.sort_values(by='Date_with_affect', inplace=True)
        self.df.drop(columns=['Date'], inplace=True)

    def filter_by_tweets(self, tweets_to_leave):
        self.df = self.df[self.df["Text"].isin(tweets_to_leave)]  # .copy()

    def merge_tweets_with_dollar_prices(self, dollar_prices, drop_open_and_date=True):
        dollar_prices.sort_values(by='Date', inplace=True)  # nie sa takie same?
        self.df = pd.merge_asof(self.df, dollar_prices, left_on='Date_with_affect', right_on='Date', direction='forward')

        columns_to_drop = ['Date_with_affect', 'Open', 'Date'] if drop_open_and_date else ['Date_with_affect']
        self.df.drop(columns=columns_to_drop, inplace=True)
        # todo powinno sprawdzac czy wszystko sie dopasowało

    def set_currency_change(self):
        def _get_change(x):
            if x > up_min:
                return "Up"
            if x < down_max:
                return "Down"
            return "NC"

        down_max, up_min = calculate_thresholds(self.df) # todo to mozna do klasy wziac
        print(down_max)
        print(up_min)
        self.df["Market_change"] = self.df["Market_change"].apply(_get_change)


def calculate_thresholds(df):
    mean = df["Market_change"].mean()
    sigma = df["Market_change"].std(ddof=0)
    lower_threshold = (mean - (sigma / 3)).round(2)
    higher_threshold = (mean + (sigma / 3)).round(2)
    return lower_threshold, higher_threshold


def count_nr_of_feature_occurrences(df):
    return [(col, (df.loc[df[col] == True, col].count())) for col in df]


def get_date_to_check_affect(d):
    res = d if d.hour < 22 else d + timedelta(days=1)
    return res.normalize()

