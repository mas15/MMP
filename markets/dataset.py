import pandas as pd
from datetime import timedelta


class TweetsDataSet:
    """
    Class that wraps around a Pandas DataFrame and represents a set of tweets,
    their features, sentiments, and market effect.
    """
    non_text_feature_columns = ["Text", "Market_change", "Tweet_sentiment", "Date"]

    def __init__(self, df):
        self.df = df

    @property
    def features(self):
        return list(self.get_features_df())

    def get_features_df(self):
        return self.df.drop(columns=self.non_text_feature_columns, errors='ignore', axis=1)

    def get_no_features_df(self):
        feature_columns = [c for c in self.df.columns if c not in self.non_text_feature_columns]
        return self.df.drop(columns=feature_columns, errors='ignore')

    def save_to_csv(self, filename):
        self.df.to_csv(filename, index=False)

    def get_all_tweets(self):
        all_tweets = self.df["Text"].tolist()
        if not all_tweets:
            raise Exception("There is no tweets in the dataset")
        return all_tweets

    def set_phrase_features(self, selecting_function, features):
        for f in features:
            self.df[f] = 0
        self.df = self.df.apply(lambda row: self._mark_row(row, selecting_function), axis=1)

    def set_sentiment(self, sentiment_calc_function):
        self.df["Tweet_sentiment"] = self.df["Text"].apply(sentiment_calc_function)

    def get_sentiment(self):
        return self.df["Tweet_sentiment"]

    def set_market_change(self, change_setting_function):
        self.df["Market_change"] = self.df["Market_change"].apply(change_setting_function)

    def get_market_change(self):
        return self.df["Market_change"]

    @staticmethod
    def _mark_row(row, selecting_function):
        features = selecting_function(row['Text']).items()
        for f, is_in_tweet in features:
            if is_in_tweet:
                row[f] = 1
        return row

    def get_x(self):
        self._keep_sentiment_at_the_end()
        return self.df.drop(columns=["Text"]).values

    def get_x_y(self):
        self._keep_sentiment_at_the_end()
        y = self.df["Market_change"].values.ravel()
        x = self.df.drop(columns=["Market_change", "Text"]).values
        return x, y

    def _keep_sentiment_at_the_end(self):
        cols = list(self.df)
        cols.append(cols.pop(cols.index("Tweet_sentiment")))
        self.df = self.df.reindex(columns=cols)
        return self.df

    def get_feature_occurrences(self):
        features_with_occurrences = count_nr_of_feature_occurrences(self.get_features_df())
        return features_with_occurrences

    def _check_if_features_are_in_dataframe(self, features):
        feats_not_in_df = [f for f in features if f not in self.features]
        if feats_not_in_df:
            raise ValueError("There are {0} selected features that are not in the dataset: {1}".format(len(feats_not_in_df), feats_not_in_df))

    def remove_features(self, features_to_remove):
        self._check_if_features_are_in_dataframe(features_to_remove)
        self.df.drop(columns=features_to_remove, axis=1, inplace=True)

    def drop_instances_without_features(self):
        self.df = self.df[(self.get_features_df().T != 0).any()]

    def get_marked_features(self):
        counted = count_nr_of_feature_occurrences(self.get_features_df())
        marked = [f for f, is_marked in counted if is_marked]
        return marked

    def set_date_with_effect(self):
        self.df["Date_with_affect"] = self.df["Date"].apply(get_date_to_check_affect)
        self.df.sort_values(by='Date_with_affect', inplace=True)
        self.df.drop(columns=['Date'], inplace=True)

    def filter_by_tweets(self, tweets_to_leave):
        self.df = self.df[self.df["Text"].isin(tweets_to_leave)]

    def merge_tweets_with_dollar_prices(self, dollar_prices, drop_open_and_date=True):
        dollar_prices.sort_values(by='Date', inplace=True)
        self.df = pd.merge_asof(self.df, dollar_prices, left_on='Date_with_affect', right_on='Date', direction='forward')

        columns_to_drop = ['Date_with_affect', 'Open', 'Date'] if drop_open_and_date else ['Date_with_affect']
        self.df.drop(columns=columns_to_drop, inplace=True)


def dataset_from_text(text):
    return TweetsDataSet(pd.DataFrame({'Text': [text]}))


def count_nr_of_feature_occurrences(df):
    return [(col, (df.loc[df[col] == True, col].count())) for col in df]


def get_date_to_check_affect(d):
    res = d if d.hour < 22 else d + timedelta(days=1)
    return res.normalize()

