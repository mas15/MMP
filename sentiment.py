from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier, MaxEntClassifier
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag_sents, pos_tag
import pickle
from dataset import get_pos_and_neg_tweets_with_sentiment_from_file, get_train_and_test_data_for_k_run, split_pos_and_neg_into_folds
# from http://textblob.readthedocs.io/en/dev/classifiers.html
from nltk.corpus import stopwords

from rake_nltk import Rake
r = Rake()

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
        cl = NaiveBayesClassifier(train_data, phrases_extractor)
        accuracy = cl.accuracy(test_data)
        sum += accuracy
        print("ACCU: " + str(accuracy))
        print(cl.show_informative_features(50))
        print()
    #print("SUM: " + str(sum/NUM_OF_FOLDS))
    return cl, sum/NUM_OF_FOLDS


def preproces(tweet):
    """
    >>> t = "As a candidate, I promised we would pass a massive tax cut for the everyday, working Americans."
    >>> preproces(t)
    'candidate promised would pass massive tax cut everyday working americans'
    >>> preproces("We love you!\\n\\nGOD BLESS TEXAS & GOD BLESS THE USA")
    'love ! god bless texas & god bless usa'
    """
    # r.extract_keywords_from_text(tweet)
    # words = r.get_ranked_phrases()
    words = word_tokenize(tweet)
    words = [w.lower() for w in words]
    to_remove = stopwords.words("english") + [",", ".", "\"", "'"] + ["n't"]  # shouldn't be split
    result = [w for w in words if w not in to_remove]

    return " ".join(words)

def phrases_extractor(document):
    r.extract_keywords_from_text(document)
    words = r.get_ranked_phrases()
    words = [w.lower() for w in words]

    print()
    print(document)
    print(words)
    print()


    feats = {}
    for w in words:
        feats["contains({0})".format(w)] = True
    return feats

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
    #pos = preproces_tweet_tuples(pos)
    #neg = preproces_tweet_tuples(neg)
    print("LEN POS: " + str(len(pos)))
    print("LEN NEG: " + str(len(neg)))
    sum = 0
    for i in range(1):
        cl, aver = perform_k_fold_validation(pos, neg)
        sum += aver
    print("AV OF 1: " +str(sum))
    save_classifier(cl)
    print(cl.classify("Make america great again"))
    print(cl.classify("crooked hilary"))
    print(cl.classify("bad bad mexico"))
    print(cl.classify("As a candidate, I promised we would pass a massive tax cut for the everyday, working Americans."))


# wywalic to:
# My warmest condolences and sympathies to the victims and families of the terrible Las Vegas shooting. God bless you!
