from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier, MaxEntClassifier
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag_sents, pos_tag
import pickle
from dataset import get_pos_and_neg_tweets_with_sentiment_from_file, get_train_and_test_data_for_k_run, split_pos_and_neg_into_folds
# from http://textblob.readthedocs.io/en/dev/classifiers.html
from nltk.corpus import stopwords
from textrazor import TextRazor

TEX_RAZOR_KEY = "8286cbb1f58dbb192a6a237f8bc425502cb4cb1cd3edeef6bf7740f9"
client = TextRazor(TEX_RAZOR_KEY, extractors=["phrases"])

from gensim.models.phrases import Phraser

def get_tweet_sentiment(text):
    cl = load_classifier()
    text = preproces(text)

    text = TextBlob(text, classifier=cl)
    pol = text.sentiment.polarity
    sent = cl.classify(text)
    return sent, pol


def perform_k_fold_validation(pos, neg):
    sum = 0
    NUM_OF_FOLDS = 3
    cl = None
    pos_folds, neg_folds = split_pos_and_neg_into_folds(pos, neg, NUM_OF_FOLDS)

    for k_run in range(NUM_OF_FOLDS):
        train_data, test_data = get_train_and_test_data_for_k_run(pos_folds, neg_folds, k_run)
        cl = NaiveBayesClassifier(train_data)
        accuracy = cl.accuracy(test_data)
        sum += accuracy
        print("ACCU: " + str(accuracy))
        print(cl.show_informative_features(50))
        print()
    print("SUM: " + str(sum/NUM_OF_FOLDS))
    return cl


def preproces(tweet):
    """
    >>> t = "As a candidate, I promised we would pass a massive tax cut for the everyday, working Americans."
    >>> preproces(t)
    'candidate promised would pass massive tax cut everyday working americans'
    >>> preproces("We love you!\\n\\nGOD BLESS TEXAS & GOD BLESS THE USA")
    'love ! god bless texas & god bless usa'
    """
    words = word_tokenize(tweet)
    words = [w.lower() for w in words]
    to_remove = stopwords.words("english") + [",", ".", "\"", "'"] + ["n't"]  # shouldn't be split
    result = [w for w in words if w not in to_remove]


    response = client.analyze(text)
    for entity in response.entities():
        print
        entity

    return " ".join(result)


def preproces_tweet_tuples(tweets_with_sentiment):
    return [(preproces(t[0]), t[1]) for t in tweets_with_sentiment]


def save_classifier(cl):
    with open("classifier.pickle", "wb") as f:
        pickle.dump(cl, f)


def load_classifier():
    with open("classifier.pickle", "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    import doctest
    doctest.testmod()

    pos, neg = get_pos_and_neg_tweets_with_sentiment_from_file()
    pos = preproces_tweet_tuples(pos)
    neg = preproces_tweet_tuples(neg)
    print("LEN POS: " + str(len(pos)))
    print("LEN NEG: " + str(len(neg)))

    cl = perform_k_fold_validation(pos, neg)
    save_classifier(cl)
    print(cl.classify("Make america great again"))
    print(cl.classify("crooked hilary"))
    print(cl.classify("bad bad mexico"))
    print(cl.classify("As a candidate, I promised we would pass a massive tax cut for the everyday, working Americans."))


# wywalic to:
# My warmest condolences and sympathies to the victims and families of the terrible Las Vegas shooting. God bless you!
