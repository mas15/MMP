import pandas as pd
from datetime import timedelta


class TweetsDataSet:
    non_text_feature_columns = ["Text", "Market_change", "Tweet_sentiment", "Date"]

    def __init__(self, df):
        self.df = df

    @property
    def features(self): # TODO musza miec ciagle te same featurey w tej samej kolejnosci
        return list(self.get_features_df())

    def get_features_df(self):
        return self.df.drop(columns=self.non_text_feature_columns, errors='ignore', axis=1)

    def get_no_features_df(self):
        return self.df.drop(columns=[c for c in self.df.columns if c not in self.non_text_feature_columns], errors='ignore')

    def save_to_csv(self, filename):
        self.df.to_csv(filename, index=False)

    def get_all_tweets(self):
        all_tweets = self.df["Text"].tolist()
        if not all_tweets:
            raise Exception("There is no tweets in the dataset")
        return all_tweets

    def set_phrase_features(self, selecting_function, features):
        for f in features: # TODO musza miec ciagle te same featurey w tej samej kolejnosci !!!!
            self.df[f] = 0  # todo test ze teraz resetuje tez
        self.df = self.df.apply(lambda row: self._mark_row(row, selecting_function), axis=1)
        self.df.fillna(0, inplace=True) # todo remove

    def set_sentiment(self, sentiment_calc_function):
        self.df["Tweet_sentiment"] = self.df["Text"].apply(sentiment_calc_function)

    def get_sentiment(self):
        return self.df["Tweet_sentiment"]

    def set_market_change(self, change_setting_function):
        self.df["Market_change"] = self.df["Market_change"].apply(change_setting_function)

    def get_market_change(self): # todo rename
        return self.df["Market_change"]

    @staticmethod
    def _mark_row(row, selecting_function):
        features = selecting_function(row['Text']).items()
        for f, is_in_tweet in features:
            if is_in_tweet:
                row[f] = 1
        return row

    def get_x(self):
        return self.df.drop(columns=["Text"]).values  # todo get features df with sentiment?

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
        self.df = self.df[self.df["Text"].isin(tweets_to_leave)]  # .copy()

    def merge_tweets_with_dollar_prices(self, dollar_prices, drop_open_and_date=True):
        dollar_prices.sort_values(by='Date', inplace=True)  # nie sa takie same?
        self.df = pd.merge_asof(self.df, dollar_prices, left_on='Date_with_affect', right_on='Date', direction='forward')

        columns_to_drop = ['Date_with_affect', 'Open', 'Date'] if drop_open_and_date else ['Date_with_affect']
        self.df.drop(columns=columns_to_drop, inplace=True)
        # todo powinno sprawdzac czy wszystko sie dopasowało



def count_nr_of_feature_occurrences(df):
    return [(col, (df.loc[df[col] == True, col].count())) for col in df]


def get_date_to_check_affect(d):
    res = d if d.hour < 22 else d + timedelta(days=1)
    return res.normalize()

