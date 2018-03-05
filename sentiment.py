from nltk import NaiveBayesClassifier
import pickle
from dataset import *
import nltk
from feature_extractor import FeatureExtractor


class SentimentAnalyser:
    def __init__(self):
        self.extr = FeatureExtractor()
        self.cl = None

    def save(self):
        with open("classifier.pickle", "wb") as f:
            pickle.dump(self.cl, f)
            pickle.dump(self.extr, f)

    def load(self):
        with open("classifier.pickle", "rb") as f:
            self.cl = pickle.load(f)
            self.extr = pickle.load(f)

    def train(self, train_data):
        self.extr.build_vocabulary(train_data)
        training_features = nltk.classify.apply_features(self.extr.extract_features, train_data)  # labeled = True
        self.cl = NaiveBayesClassifier.train(training_features)

    def check_accuracy(self, test_data):
        testing_features = nltk.classify.apply_features(self.extr.extract_features, test_data)
        return nltk.classify.accuracy(self.cl, testing_features)

    def analyse(self, tweet):
        problem_features = self.extr.extract_features(tweet)
        return self.cl.classify(problem_features)

    def run_k_fold(self, pos, neg, nr_folds=5):
        sum = 0
        pos_folds, neg_folds = split_pos_and_neg_into_folds(pos, neg, nr_folds)

        for k_run in range(nr_folds):
            train_data, test_data = get_train_and_test_data_for_k_run(pos_folds, neg_folds, k_run)
            self.train(train_data)
            accuracy = self.check_accuracy(test_data)

            sum += accuracy
        print("ACCU: " + str(sum/nr_folds))
        print(self.cl.show_most_informative_features(20))
        print()
        return sum / nr_folds

    def train_on_all(self, pos, neg):
        self.train(pos + neg)


if __name__ == "__main__":
    pos, neg = get_pos_and_neg_tweets_with_sentiment_from_file()
    sent = SentimentAnalyser()

    # TODO min 3 litery ale vez kropki, We love you!\n\nGOD  nie dobre /n

    sum = 0
    for i in range(40):
        sum += sent.run_k_fold(pos, neg)
    print("VOCABULARY:")
    print(sent.extr.vocabulary)
    print(sent.extr.phrases)
    print("AV OF 40: " + str(sum / 40))

    print("TRAINING ON ALL")
    sent.train_on_all(pos, neg)
    sent.save()

    # print(sent.analyse("Make america great again"))
    # print("TUTEJ _-------------------------------------")
    # d = sent.extr.extract_features("Make america great again")
    # print([f for f, v in d.items() if v])
    #
    # print(sent.analyse("bad bad mexico"))
    # print([f for f, v in sent.extr.extract_features("bad bad mexico").items() if v])
    #
    # print(sent.analyse("As a candidate, I promised we would pass a massive tax cut for the everyday, working Americans."))
    # print([f for f, v in sent.extr.extract_features(
    #     "As a candidate, I promised we would pass a massive tax cut for the everyday, working Americans.").items() if
    #        v])










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
