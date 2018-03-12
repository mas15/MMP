from rake import Rake, load_stop_words
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class FeatureExtractor:
    def __init__(self):
        self.lemamatizer = WordNetLemmatizer()
        self.r = Rake("SmartStoplist.txt", 3, 3, 2)  # todo
        """ Features set containing unique words"""
        self.vocabulary = set()
        self.phrases = []
        self.words_to_remove = load_stop_words("SmartStoplist.txt") + ["n't", "'s", "...", "w/",
                                                                       ]  # shouldn't be split
        self.words_to_remove += list("!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~0123456789’")

    def extract_features(self, tweet):
        features = {}
        for phrase in self.phrases:
            features[phrase] = (phrase in tweet.lower())
            tweet = tweet.replace(phrase, "")  # TODO not sure if not spoil tokenizing

        words_in_tweet = self.extract_words_from_tweet(tweet)
        for word in self.vocabulary:
            features[word] = (word in words_in_tweet)
        return features

    def extract_words_from_tweet(self, tweet):
        def _to_skip(w):
            return w in self.words_to_remove or w.replace(".", "").isdigit()

        def _clear(w):
            return w.replace("`", "").replace("'", "").replace("\"", "").replace("\n", "").replace("\\", "")

        words = word_tokenize(tweet)
        words = [w.lower() for w in words]
        words = [w for w in words if not _to_skip(w)]
        words = [_clear(w) for w in words]
        words = [self.lemamatizer.lemmatize(w) for w in words if w]
        return words

    def build_vocabulary(self, dataset):
        all_words = []

        all_tweets_contents = " ".join([t for t, s in dataset])
        phrases = self.r.run(all_tweets_contents)  # todo czy one juz lower sa?
        # print("PHRASES")
        # for x in self.phrases:
        #     if " " in x[0]:
        #         print(x)

        self.phrases = [phrase for phrase, score in phrases if " " in phrase]

        for tweet, s in dataset:
            all_words += self.extract_words_from_tweet(tweet)
        self.vocabulary = list(set(all_words))
        # print()
        # print("LEN OF VOCABULARY:" + str(len(self.vocabulary)))
        # print()
