from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier, DecisionTreeClassifier
import csv
from sklearn.model_selection import KFold
import numpy as np
from random import shuffle
from math import ceil


# from http://textblob.readthedocs.io/en/dev/classifiers.html

def get_tweets_with_sentiment():
    pos = []
    neg = []
    with open('aaa.csv', 'r', encoding='utf8') as f:
        reader = csv.reader(f, delimiter=",")
        try:
            for line in reader:
                content, sentiment = line[1], line[3]
                if sentiment == "pos":
                    pos.append((content, sentiment))
                # elif sentiment != "neg":
                #     print(content)
                #     print(sentiment)
                else:
                    neg.append((content, sentiment))
        except IndexError:
            pass
        return pos, neg


def get_tweet_sentiment(text):
    pos, neg = get_tweets_with_sentiment()
    cl = NaiveBayesClassifier(pos + neg)
    blob = TextBlob(text, classifier=cl)
    sent = blob.classify()
    pol = blob.polarity
    return sent, pol


def get_train_and_test(folds, k_run):
    """
    >>> get_train_and_test([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10]], 1)
    ([5, 6, 7, 8, 9, 10], [1, 2, 3, 4])
    >>> get_train_and_test([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10]], 3)
    ([1, 2, 3, 4, 5, 6, 7, 8], [9, 10])
    """
    train_data = folds[:]
    test_data = train_data.pop(k_run - 1)
    train_data = [item for fold_items in train_data for item in fold_items]
    return train_data, test_data


def split_pos_and_neg_into_folds(pos, neg, n, random=True):
    """
    >>> split_pos_and_neg_into_folds(range(1,11), range(21, 31), 3, False)
    ([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10]], [[21, 22, 23, 24], [25, 26, 27, 28], [29, 30]])
    """
    pos_folds, neg_folds = [], []
    set_size = len(pos)
    fold_size = ceil(set_size / n)
    indices = list(range(0, set_size))
    if random:
        shuffle(indices)
    while indices:
        fold_indices, indices = indices[:fold_size], indices[fold_size:]
        pos_folds.append([pos[x] for x in fold_indices])
        neg_folds.append([neg[x] for x in fold_indices])
    return pos_folds, neg_folds


def get_train_and_test_data_for_k_run(pos_folds, neg_folds, k_run):
    """
    >>> p = [[(1,1)], [(2,2)], [(3,3)]]
    >>> n = [[(4,4)], [(4,4)], [(4,4)]]
    >>> get_train_and_test_data_for_k_run(p, n, 1)
    ([(2, 2), (3, 3), (4, 4), (4, 4)], [(1, 1), (4, 4)])
    """
    pos_train, pos_test = get_train_and_test(pos_folds, k_run)
    neg_train, neg_test = get_train_and_test(neg_folds, k_run)
    return pos_train + neg_train, pos_test + neg_test


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    pos, neg = get_tweets_with_sentiment()
    print("LEN POS: " + str(len(pos)))
    print("LEN NEG: " + str(len(neg)))

    sum = 0
    NUM_OF_FOLDS = 3
    cl = None
    pos_folds, neg_folds = split_pos_and_neg_into_folds(pos, neg, NUM_OF_FOLDS)

    for k_run in range(NUM_OF_FOLDS):
        train_data, test_data = get_train_and_test_data_for_k_run(pos_folds, neg_folds, k_run)
        cl = NaiveBayesClassifier(train_data)
        sum += cl.accuracy(test_data)

    average = sum / 3
    print("AVERGAE: " + str(average))
    print(cl.show_informative_features(20))

    print()
    blob = TextBlob("Make america great again", classifier=cl)
    print(blob.classify())
    print(blob.polarity)
    prob_dist = cl.prob_classify("Make america great again")
    print((prob_dist.prob('pos'), prob_dist.prob('neg')))

    print()
    print()
    blob = TextBlob("crooked Hilary", classifier=cl)
    print(blob.classify())
    print(blob.polarity)
    prob_dist = cl.prob_classify("crooked Hilary")
    print((prob_dist.prob('pos'), prob_dist.prob('neg')))
