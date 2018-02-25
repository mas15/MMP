from textblob import TextBlob
# from textblob.classifiers import NaiveBayesClassifier, MaxEntClassifier
from nltk import NaiveBayesClassifier
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag_sents, pos_tag
import pickle
from dataset import get_pos_and_neg_tweets_with_sentiment_from_file, get_train_and_test_data_for_k_run, split_pos_and_neg_into_folds
# from http://textblob.readthedocs.io/en/dev/classifiers.html
from nltk.corpus import stopwords

import nltk

from rake import Rake, load_stop_words
r = Rake("SmartStoplist.txt", 3, 3, 2)

from nltk.stem import WordNetLemmatizer
ps = WordNetLemmatizer()
RESULT = set()


class FeatureExtractor:
    def __init__(self):
        """ Features set containing unique words"""
        self.vocabulary = set()
        self.phrases = []
        self.words_to_remove = load_stop_words("SmartStoplist.txt") + ["n't", "'s", "..."]  # shouldn't be split
        self.words_to_remove += list("!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~0123456789")

    def extract_features(self, tweet):
        words_in_tweet = self.extract_words_from_tweet(tweet)
        features = {}
        for word in self.vocabulary:
            features[word] = (word in words_in_tweet)
        for phrase in self.phrases:
            features[phrase] = (phrase in words_in_tweet)
        return features

    def extract_words_from_tweet(self, tweet):
        # words_with_score = r.run(tweet)
        # words_with_score = [w for w in words_with_score if w[1] < 5]
        [tweet.replace(p, "") for p in self.phrases]  # TODO not sure if not spoil tokenizing
        words = word_tokenize(tweet)
        words = [w.lower() for w in words]
        words = [w for w in words if w not in self.words_to_remove]

        for w in words:
            if w != ps.lemmatize(w):
                RESULT.add((w, ps.lemmatize(w)))

        words = [ps.lemmatize(w) for w in words]
        return words

    def build_vocabulary(self, dataset):
        all_words = []

        add_tweets_contents = " ".join([t for t, s in dataset])
        phrases = r.run(add_tweets_contents)
        # print("PHRASES")
        # for x in self.phrases:
        #     if " " in x[0]:
        #         print(x)

        self.phrases = [phrase for phrase, score in phrases]
        for tweet, s in dataset:
            all_words += self.extract_words_from_tweet(tweet)
        self.vocabulary = list(set(all_words))
        print()
        print("LEN OF VOCABULARY:" + str(len(self.vocabulary)))
        print()


extr = FeatureExtractor()


def get_tweet_sentiment(text):
    cl = load_classifier()
    problem_features = extr.extract_features(text)
    sent = cl.prob_classify(problem_features)
    pol = "nie wiem co tutaj"
    return sent, pol


def train_classifier(train_data):
    # todo tutaj jakoś obiekt tworzyć
    extr.build_vocabulary(train_data)

    training_features = nltk.classify.apply_features(extr.extract_features, train_data) #labeled = True
    cl = NaiveBayesClassifier.train(training_features)
    return cl


def check_classifier_accuracy(cl, test_data):
    testing_features = nltk.classify.apply_features(extr.extract_features, test_data)
    return nltk.classify.accuracy(cl, testing_features)


def perform_k_fold_validation(pos, neg):
    sum = 0
    NUM_OF_FOLDS = 3
    pos_folds, neg_folds = split_pos_and_neg_into_folds(pos, neg, NUM_OF_FOLDS)

    for k_run in range(NUM_OF_FOLDS):
        train_data, test_data = get_train_and_test_data_for_k_run(pos_folds, neg_folds, k_run)

        cl = train_classifier(train_data)
        accuracy = check_classifier_accuracy(cl, test_data)

        sum += accuracy
        print("ACCU: " + str(accuracy))
        print(cl.show_most_informative_features(10))
        print()
    #print("SUM: " + str(sum/NUM_OF_FOLDS))
    return cl, sum/NUM_OF_FOLDS







def save_classifier(cl):
    with open("classifier.pickle", "wb") as f:
        pickle.dump(cl, f)


def load_classifier():
    with open("classifier.pickle", "rb") as f:
        return pickle.load(f)



if __name__ == "__main__":
    #import doctest
    #doctest.testmod()

    pos, neg = get_pos_and_neg_tweets_with_sentiment_from_file()

    sum = 0
    for i in range(40):
        cl, aver = perform_k_fold_validation(pos, neg)
        sum += aver
    print("AV OF 40: " + str(sum/40))
    save_classifier(cl)



    print(cl.classify(extr.extract_features("Make america great again")))
    print([f for f, k in extr.extract_features("Make america great again") if k])
    print(cl.classify(extr.extract_features("crooked hilary")))
    print([f for f, k in extr.extract_features("crooked hilary") if k])
    print(cl.classify(extr.extract_features("bad bad mexico")))
    print([f for f, k in extr.extract_features("bad bad mexico") if k])
    print(cl.classify(extr.extract_features("As a candidate, I promised we would pass a massive tax cut for the everyday, working Americans.")))
    print([f for f, k in extr.extract_features("As a candidate, I promised we would pass a massive tax cut for the everyday, working Americans.") if k])

    print()
    for r in RESULT:
        print(r)

# wywalic to:
# My warmest condolences and sympathies to the victims and families of the terrible Las Vegas shooting. God bless you!



# def preproces(tweet):
#     """
#     >>> t = "As a candidate, I promised we would pass a massive tax cut for the everyday, working Americans."
#     >>> preproces(t)
#     'candidate promised would pass massive tax cut everyday working americans'
#     >>> preproces("We love you!\\n\\nGOD BLESS TEXAS & GOD BLESS THE USA")
#     'love ! god bless texas & god bless usa'
#     """
#     # r.extract_keywords_from_text(tweet)
#     # words = r.get_ranked_phrases()
#     words = word_tokenize(tweet)
#     words = [w.lower() for w in words]
#     to_remove = stopwords.words("english") + [",", ".", "\"", "'"] + ["n't"]  # shouldn't be split
#     result = [w for w in words if w not in to_remove]
#
#     return " ".join(words)

#
# def phrases_extractor(document):
#     #r.extract_keywords_from_text(document)
#     #x = r.get_ranked_phrases_with_scores()
#     words = r.run(document)
#     words = [w for w in words if w[1] < 5]
#     #words = [w.lower() for w in words]
#
#     print()
#     print(document)
#     print(words)
#     print()
#     words = [w[0] for w in words]
#
#     for w in words:
#         if w != ps.lemmatize(w):
#             RESULT.add((w, ps.lemmatize(w)))
#     words = [ps.lemmatize(w) for w in words]
#
#     feats = {}
#     for w in words:
#         feats["contains({0})".format(w)] = True
#     return feats
#
#
# def preproces_tweet_tuples(tweets_with_sentiment):
#     return [(preproces(t[0]), t[1]) for t in tweets_with_sentiment]
#