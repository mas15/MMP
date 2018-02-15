from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
import csv
from sklearn.model_selection import KFold
import numpy as np

# from http://textblob.readthedocs.io/en/dev/classifiers.html


def get_tweets_with_sentiment():
    pos = []
    neg = []
    with open('aaa.csv', 'r', encoding='utf8') as f:
        reader = csv.reader(f, delimiter=",")
        try:
            for line in reader:
                print(line)
                content, sentiment = line[1], line[3]
                if sentiment == "pos":
                    pos.append((content, sentiment))
                else:
                    neg.append((content, sentiment))
        except IndexError:
            pass
        return pos, neg


if __name__ == "__main__":
    pos, neg = get_tweets_with_sentiment()
    # pos_test_set, pos_train_set = pos[25:], pos[:13]
    # neg_test_set, neg_train_set = pos[25:], pos[:13]

    # train_set = pos_train_set + neg_train_set
    # test_set = pos_test_set + neg_test_set

    # cl = NaiveBayesClassifier(train_set)
    # print()
    # print(cl.accuracy(test_set))
    # print(cl.classify("make america great again!"))
    # print(cl.classify("that is sad!"))
    # cl.show_informative_features(20)

    print(len(pos))
    print(len(neg))

    kf = KFold(n_splits=3)
    sum = 0
    for train_ind, test_ind in kf.split(pos): # random.sample(range(100), 10)
        train_data, test_data = [], []
        for i in range(len(pos)):
            if i in train_ind:
                train_data.append(pos[i])
                train_data.append(neg[i])
            else:
                test_data.append(pos[i])
                test_data.append(neg[i])

        cl = NaiveBayesClassifier(train_data)
        sum += cl.accuracy(test_data)

    average = sum / 3
    print(average)
