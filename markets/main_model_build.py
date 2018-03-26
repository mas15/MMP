from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import pandas as pd
import numpy as np
import pickle
from markets.feature_extractor import FeatureExtractor
from markets.sentiment import SentimentAnalyser
from markets.association import mark_features, calculate_sentiment
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

ASSOCIATION_MODEL_FILE = os.path.join(os.path.dirname(__file__), "assoc_model.pickle")
FEATURES_WITH_EFFECT_FILE = os.path.join(os.path.dirname(__file__), "data/features_with_effect.csv")
FEATURES_WITH_TEXT_AND_EFFECT_FILE = os.path.join(os.path.dirname(__file__), "data/text_with_feats_and_effect.csv")
pd.set_option('display.width', 1500)

MIN_FEATURE_OCCURENCIES = 6


class PredictingModel:
    def __init__(self):
        self.model = None
        self.extr = FeatureExtractor()
        self.features = []
        self.sent = SentimentAnalyser() # todo ogarnac to
        self.sent.load()

    def build_model(self):
        df = pd.read_csv(FEATURES_WITH_TEXT_AND_EFFECT_FILE)
        df.drop(columns=['Text'], inplace=True) # TODO zachowac jakos
        y = df['Market_change'].values
        df = df.drop(columns=['Market_change'])
        x = df.values

        self.features = df.columns.tolist()
        self.extr.set_features(df.columns)

        # x = SelectKBest(chi2, k=117).fit_transform(x, y)

        sum_train, sum_test = 0, 0

        # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=69)

        # misclassified_objects = {i: 0 for i in range(1, len(x) + 1)}

        best_model_accu = 0
        for i in range(1, 31):
            print(i)
            run_sum_train, run_sum_test = 0, 0
            kf = StratifiedKFold(n_splits=10, random_state=i, shuffle=True)
            for train_index, test_index in kf.split(x, y):
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]

                #model = MultinomialNB()
                model = LogisticRegressionCV(random_state=123, cv=10, Cs=3)
                model.fit(x_train, y_train.ravel())

                accu_on_train, misclass_on_train = self.test_model_on_dataset(model, x_train, y_train)
                accu_on_test, misclass_on_test = self.test_model_on_dataset(model, x_test, y_test)

                run_sum_train += accu_on_train
                run_sum_test += accu_on_test

                if accu_on_test > best_model_accu:
                    self.model = model

            sum_train += run_sum_train / 10
            sum_test += run_sum_test / 10

        print()
        print("Accuracy on train: {0}".format(sum_train / 30))
        print("Accuracy on test:  {0}".format(sum_test / 30))
        print()

    def test_model_on_dataset(self, model, x, y):
        predicted = model.predict(x)
        accuracy = accuracy_score(y, predicted)
        misclassified_objects = np.where(y != predicted)
        return accuracy, misclassified_objects

    def analyse(self, text): # todo co jak nie ma modelu
        features = self.extract_features(text)
        features.drop(columns=["Text"], inplace=True)
        prediction, propabs = self.predict(features)

        result = put_results_in_dict(prediction, propabs, features)
        return result

    def predict(self, features):
        result = self.model.predict(features)
        result = str(result[0])
        propabs_vals = self.model.predict_proba(features)
        propabs_vals = propabs_vals[0].tolist()
        propabs = dict(zip(self.model.classes_, propabs_vals))
        return result, propabs

    def extract_features(self, text): # todo get_features_vector?
        df = pd.DataFrame({'Text': [text]})
        df = mark_features(self.extr, df)
        df = calculate_sentiment(df, self.sent)
        # extract features and sentiment
        return df

    def save(self):
        with open(ASSOCIATION_MODEL_FILE, "wb") as f:
            pickle.dump(self.model, f)
            pickle.dump(self.extr,  f)
            pickle.dump(self.features,  f)

    def load(self):
        with open(ASSOCIATION_MODEL_FILE, "rb") as f:
            self.model = pickle.load(f)
            self.extr = pickle.load(f)
            self.features = pickle.load(f)

    def get_most_coefficient_features(self):
        result = dict()
        for i, target in enumerate(self.model.classes_): # todo check if nr feats = coef
            feats = sorted(zip(self.features, self.model.coef_[i]), key=lambda t: t[1])
            result[target] = feats
        return result


def put_results_in_dict(prediction, propabs, features): # todo przeniesc]
    from markets.association import drop_infrequent_features
    result = dict(propabs)
    result["prediction"] = prediction
    sentiment_value = features["Tweet_sentiment"].iloc[0]
    features.drop(columns="Tweet_sentiment", inplace=True)
    found_features = drop_infrequent_features(features, 1)
    result["features"] = found_features.columns.tolist()
    result["sentiment"] = "Possitive" if sentiment_value else "Negative"
    return result


if __name__ == '__main__':
    model = PredictingModel()
    model.build_model()
    model.save()
    model.load()
    print(model.get_most_coefficient_features())
    print(model.analyse("Bad bad Mexicans")) # todo nie przewiduje po zmienionym tsh
