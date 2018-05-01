"""
Module containing a class responsible for tweets sentiment analysis. It wraps NaiveBayesClassifier
from NLTK library and uses a PhrasesExtractor to extract features from tweets which are then used to
train a model or predict a value of the particular tweet. All of the functionality was wrapped
in a class because it is more convenient to load and save the Analyser and perform any tests.
"""
import os
import pickle
import nltk
import csv
from nltk import NaiveBayesClassifier
from markets.utils import get_x_y_from_list_of_tuples, k_split
from markets.phrases_extraction import PhrasesExtractor

SENTIMENT_MODEL_FILE = os.path.join(os.path.dirname(__file__), "pickled_models/sentiment_model.pickle")
CORPUS_FILE = os.path.join(os.path.dirname(__file__), "data/sentimental_tweets.csv")


class SentimentAnalyser:
    def __init__(self, extr=None, cl=None):
        self.extr = extr or PhrasesExtractor()
        self.cl = cl

    def save(self):
        with open(SENTIMENT_MODEL_FILE, "wb") as f:
            pickle.dump(self.cl, f)
            pickle.dump(self.extr, f)

    def load(self):
        with open(SENTIMENT_MODEL_FILE, "rb") as f:
            self.cl = pickle.load(f)
            self.extr = pickle.load(f)

    def train(self, train_data):
        only_tweets = [t for t, s in train_data]
        self.extr.build(only_tweets)
        training_features = nltk.classify.apply_features(self.extr.extract_features, train_data)
        self.cl = NaiveBayesClassifier.train(training_features)

    def check_accuracy(self, test_data):
        testing_features = nltk.classify.apply_features(self.extr.extract_features, test_data)
        return nltk.classify.accuracy(self.cl, testing_features)

    def predict(self, tweet):
        problem_features = self.extr.extract_features(tweet)
        return self.cl.classify(problem_features)

    def predict_score(self, tweet):
        problem_features = self.extr.extract_features(tweet)
        prob_res = self.cl.prob_classify(problem_features)
        result = prob_res.prob("pos")
        return result

    def cross_validate(self, dataset, random_state, nr_folds=5):
        sum_in_runs = 0
        x, y = get_x_y_from_list_of_tuples(dataset)

        for x_train, x_test, y_train, y_test in k_split(x, y, nr_folds, random_state):
            train_data = list(zip(x_train, y_train))
            test_data = list(zip(x_test, y_test))

            self.train(train_data)
            accuracy = self.check_accuracy(test_data)
            sum_in_runs += accuracy

        # print("Accuracy: " + str(sum_in_runs / nr_folds))
        # print(self.cl.show_most_informative_features(20))
        return sum_in_runs / nr_folds


def get_tweets_with_sentiment_from_file(filename):
    result = []
    with open(filename, encoding='utf8') as f:
        reader = csv.reader(f, delimiter=",")
        try:
            for line in reader:
                _, content, _, sentiment = line
                if sentiment != "neg" and sentiment != "pos":
                    raise Exception("Error while reading sentiment in line: {0}".format(line))  # TODO test it
                result.append((content, sentiment))
        except IndexError:
            pass
        return result


if __name__ == "__main__":
    tweets_with_sent = get_tweets_with_sentiment_from_file(CORPUS_FILE)
    sent = SentimentAnalyser()

    average_of_n = sum([sent.cross_validate(tweets_with_sent, i) for i in range(40)])/40
    print("VOCABULARY:")
    print(sent.extr._vocabulary)
    print(sent.extr._phrases)
    print("AV OF 40: " + str(average_of_n))

    print("TRAINING ON ALL AND SAVING CL")
    sent.train(tweets_with_sent)
    sent.save()

    sent.load()
    print(sent.predict_score("Make america great again"))
    print(sent.predict_score("Bad Mexico"))
    print(sent.cl.show_most_informative_features(50))