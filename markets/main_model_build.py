from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import pandas as pd
import pickle

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

ASSOCIATION_MODEL_FILE = os.path.join(os.path.dirname(__file__), "assoc_model.pickle")
FEATURES_WITH_EFFECT_FILE = os.path.join(os.path.dirname(__file__), "data/features_with_effect.csv")
pd.set_option('display.width', 1500)

MIN_FEATURE_OCCURENCIES = 6


def build_model():
    df = pd.read_csv(FEATURES_WITH_EFFECT_FILE)
    y = df['Change'].values
    df = df.drop(columns=['Change'])
    x = df.values



    x = SelectKBest(chi2, k=117).fit_transform(x, y)


    sum_train, sum_test = 0, 0

    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=69)

    kf = KFold(n_splits=10, random_state=123)
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = MultinomialNB()
        #model = LogisticRegressionCV(random_state=123, cv=10, Cs=3)
        model.fit(x_train, y_train.ravel())

        accuracy_on_train = accuracy_score(y_train, model.predict(x_train))
        accuracy_on_test = accuracy_score(y_test, model.predict(x_test))

        sum_train += accuracy_on_train
        sum_test += accuracy_on_test

    print()
    print("Accuracy on train: {0}".format(sum_train /10))
    print("Accuracy on test:  {0}".format(sum_test / 10))
    print()
    return model

# TODO remove empty rows
# TODO add st dev as a treshold
def save_model(model):
    with open(ASSOCIATION_MODEL_FILE, "wb") as f:
        pickle.dump(model, f)


def load_model():
    with open(ASSOCIATION_MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    return model


if __name__ == '__main__':
    model = build_model()
    #save_model(model)
    #model = load_model()
    #print(model.predict("Bad Mexicans and taxes"))