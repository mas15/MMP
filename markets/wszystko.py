from datetime import timedelta
import pandas as pd
import os
from markets.sentiment import SentimentAnalyser
from markets.feature_extractor import FeatureExtractor
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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


def get_change(x, t):
    if x > t:
        return "Up"
    elif x < -t:
        return "Down"
    else:
        return "NC"


if __name__ == "__main__":
    all_tweets = read_all_tweets()
    dollar_prices = read_dollar_prices()
    sent = SentimentAnalyser()
    sent.load()

    all_tweets["Date_with_affect"] = all_tweets["Date"].apply(get_date_to_check_affect)
    all_tweets.sort_values(by='Date_with_affect', inplace=True)
    all_tweets.drop(columns=['Id', 'Date'], inplace=True)

    dollar_prices.sort_values(by='Date', inplace=True)  # nie sa takie same?
    # Text, Date with affect

    result = pd.merge_asof(all_tweets, dollar_prices, left_on='Date_with_affect', right_on='Date', direction='forward')

    result.to_csv("tweets_with_prices.csv", index=False)
    result.drop(columns=['Price', 'Open', 'High', 'Low', 'Date', 'Date_with_affect'], inplace=True)

    result["Sentiment"] = result["Text"].apply(sent.analyse)
    result["Sentiment"] = result["Sentiment"].replace({"pos": 1, "neg": 0})
    print("Sentiment calculated")

    extr = FeatureExtractor(min_keyword_frequency=4)
    extr.build_vocabulary(result["Text"].tolist())
    for f in extr.features:
        result[f] = 0

    # move mark which features are in a tweet
    result = result.apply(lambda x: mark_features(x, extr), axis=1)
    print("Features marked")


    # cols_to_leave = [line.strip() for line in open("data/attr_after_6_wr_nb_bf", 'r')]  # todo to policzyc jeszcze raz
    # cols_to_leave += ["Sentiment", "Text", "Change"]
    # cols_to_drop = [c for c in list(result) if c not in cols_to_leave]
    # result.drop(columns=cols_to_drop, axis=1, inplace=True)

    t = 1
    best = 0
    best_t = 0
    result_2 = result.copy()
    while t < 3:
        result = result_2.copy()

        features = result.drop(columns=["Sentiment", "Change", "Text"])
        cols_with_nr_of_trues = [(col, (features.loc[features[col] == True, col].count())) for col in features]
        cols_with_nr_of_trues.sort(key=lambda t: t[1])
        cols_to_drop = [c[0] for c in cols_with_nr_of_trues if c[1] <= t]
        print("Dropping " + str(len(cols_to_drop)))
        print(cols_to_drop)
        result.drop(columns=cols_to_drop, axis=1, inplace=True)

        # Change the change into True/False (dollar up, down)
        result["Dollar_up"] = result["Change"].apply(get_change, args=(0.05,))
        result.drop(columns=['Text', 'Change'], inplace=True)
        #print(result.head())

        print("Nr of NC: " + str(result.loc[result["Dollar_up"] == "NC", "Dollar_up"].count()))
        u = result.loc[result["Dollar_up"] == "Up", "Dollar_up"].count()
        d = result.loc[result["Dollar_up"] == "Down", "Dollar_up"].count()
        # u = u if u>d else d
        # zeroR = u/result["Dollar_up"].count()
        # print("zeroR: " + str(zeroR))


        y = result['Dollar_up'].values
        result = result.drop(columns=['Dollar_up'])
        x = result.values


        sum_train, sum_test = 0, 0

        kf = KFold(n_splits=10, random_state=123)
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # nb_model = GaussianNB()
            nb_model = LogisticRegressionCV(random_state=123, cv=10, Cs=3)
            nb_model.fit(x_train, y_train.ravel())

            accuracy_on_train = accuracy_score(y_train, nb_model.predict(x_train))
            accuracy_on_test = accuracy_score(y_test, nb_model.predict(x_test))

            sum_train += accuracy_on_train
            sum_test += accuracy_on_test

        print()
        print(t)
        print(sum_train /10)
        accu = sum_test / 10
        print(accu)
        print()
        if accu > best:
            best = accu
            best_t = t

        t+=1

    print()
    print()
    print("BEST")
    print(best_t)
    print(best)