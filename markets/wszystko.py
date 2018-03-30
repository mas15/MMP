from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LassoCV
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score
import os
import pandas as pd
import pickle
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif

ASSOCIATION_MODEL_FILE = os.path.join(os.path.dirname(__file__), "assoc_model.pickle")
FEATURES_WITH_EFFECT_FILE = os.path.join(os.path.dirname(__file__), "data/features_with_effect.csv")
pd.set_option('display.width', 1500)

if __name__ == "__main__":
    df = pd.read_csv(FEATURES_WITH_EFFECT_FILE)
    y = df['Market_change'].values
    df = df.drop(columns=['Market_change'])
    x = df.values

    model = MultinomialNB()

    # sfs1 = SFS(model,
    #            k_features=(50, 150),
    #            forward=True,
    #            n_jobs=-1,
    #            floating=False,
    #            verbose=2,
    #            scoring='accuracy',
    #            cv=0)
    #
    # sfs1 = sfs1.fit(x, y.ravel())
    #
    # print('Best accuracy score: %.2f' % sfs1.k_feature_idx_)
    # print('Best subset:', sfs1.k_score_)

    selector = RFECV(model, 1, cv=10, verbose=2, n_jobs=-1)
    selector = selector.fit(x, y.ravel())
    print(selector.support_)
    print(selector.ranking_)

    wybrane = []
    for i, value in enumerate(selector.ranking_):
        if value < 50:
            wybrane.append(i)



    print("wybrane")
    print(wybrane)
    features_names = df.columns.tolist()
    selected_features = [features_names[i] for i in wybrane]
    for f in selected_features:
        print(f)
#
#
#
#
#
# # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=69)
#
# best = 0
# best_i = 0
# X = x
# for i in range(10, 300):
#     # mod = LassoCV().fit(X, y)
#     # selector = SelectFromModel(mod, prefit=True)
#     # x = selector.transform(X)
#
#     selector = SelectKBest(chi2, k=i)
#     x = selector.fit_transform(X, y)
#
#     sum_train, sum_test = 0, 0
#     kf = KFold(n_splits=10, random_state=123)
#     for train_index, test_index in kf.split(x):
#         x_train, x_test = x[train_index], x[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#
#         #model = MultinomialNB()
#         model = LogisticRegressionCV(random_state=123, cv=10, Cs=3)
#         model.fit(x_train, y_train.ravel())
#
#         accuracy_on_train = accuracy_score(y_train, model.predict(x_train))
#         accuracy_on_test = accuracy_score(y_test, model.predict(x_test))
#
#         sum_train += accuracy_on_train
#         sum_test += accuracy_on_test
#
#     if (sum_test / 10) > best:
#         best_s = selector
#         best = sum_test / 10
#         best_i = i
#         print("Accuracy on train: {0}".format(sum_train / 10))
#         print("Accuracy on test:  {0}".format(sum_test / 10))
#         print()
#
#
# print()
# print(best_i)
# print(best)
# print()
# for f, v in zip(list(df), list(selector.get_support())):
#     if v:
#         print(f)
