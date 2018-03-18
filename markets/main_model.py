from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import pandas as pd

FEATURES_WITH_EFFECT_FILE = os.path.join(os.path.dirname(__file__), "data/features_with_effect.csv")
pd.set_option('display.width', 1500)

df = pd.read_csv(FEATURES_WITH_EFFECT_FILE)
MIN_FEATURE_OCCURENCIES = 6
#
# features = df.drop(columns=["Dollar up", "Sentiment"])
# cols_with_nr_of_trues = [(col, (features.loc[features[col] == True, col].count())) for col in features]
# cols_with_nr_of_trues.sort(key=lambda t: t[1])
# cols_to_drop = [c[0] for c in cols_with_nr_of_trues if c[1] <= MIN_FEATURE_OCCURENCIES]
# df.drop(columns=cols_to_drop)


df["Sentiment"] = df["Sentiment"].replace({"pos": 1, "neg": 0})
df = df.astype(int)
y = df['Dollar_up'].values
df = df.drop(columns=['Dollar_up'])
x = df.values

sum_train, sum_test = 0, 0

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=69)

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
print(sum_train /10)
print(sum_test / 10)
print()
