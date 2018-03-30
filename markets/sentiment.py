import pickle
import nltk
import os
from nltk import NaiveBayesClassifier
from markets.helpers import split_pos_and_neg_into_folds, get_train_and_test_data_for_k_run, \
    get_pos_and_neg_tweets_with_sentiment_from_file
from markets.feature_extractor import FeatureExtractor

SENTIMENT_MODEL_FILE = os.path.join(os.path.dirname(__file__), "pickled_models/sentiment_model.pickle")


class SentimentAnalyser:
    def __init__(self):
        self.extr = FeatureExtractor()
        self.cl = None

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
        self.extr.build_vocabulary(only_tweets)
        training_features = nltk.classify.apply_features(self.extr.extract_features, train_data)  # labeled = True
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
        # result = map_prob_to_score(prob_res.prob("pos"))
        result = prob_res.prob("pos")
        return result

    def cross_validate(self, pos, neg, nr_folds=5):
        sum = 0
        pos_folds, neg_folds = split_pos_and_neg_into_folds(pos, neg, nr_folds)

        for k_run in range(nr_folds):
            train_data, test_data = get_train_and_test_data_for_k_run(pos_folds, neg_folds, k_run)
            self.train(train_data)
            accuracy = self.check_accuracy(test_data)

            sum += accuracy
        print("Accuracy: " + str(sum / nr_folds))
        print(self.cl.show_most_informative_features(20))
        return sum / nr_folds

    def train_on_all(self, pos, neg):
        self.train(pos + neg)


def map_prob_to_score(prob):
    """
    Map from 0.0 - 1.0 to -1 - 1
    >>> map_prob_to_score(0.92)
    0.84
    >>> map_prob_to_score(0.5)
    0.0
    """
    return round(prob * 2 - 1, 2)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    pos, neg = get_pos_and_neg_tweets_with_sentiment_from_file()
    sent = SentimentAnalyser()

    # # todo lemmatize phrases
    # # TODO min 3 litery ale vez kropki, We love you!\n\nGOD  nie dobre /n
    # sum = 0
    # for i in range(40):
    #     sum += sent.run_k_fold(pos, neg)
    # print("VOCABULARY:")
    # print(sent.extr._vocabulary)
    # print(sent.extr._phrases)
    # print("AV OF 40: " + str(sum / 40))

    print("TRAINING ON ALL AND SAVING CL")
    sent.train_on_all(pos, neg)
    sent.save()

    sent.load()
    print(sent.predict_score("Make america great again"))
    print(sent.predict_score("Bad Mexico"))
