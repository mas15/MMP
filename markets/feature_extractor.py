from nltk.stem import WordNetLemmatizer
import re
import string
import os

STOP_LIST_FILE = os.path.join(os.path.dirname(__file__), "data/SmartStoplist.txt")
punct_remove_translator = str.maketrans('', '', string.punctuation)


def load_stop_words(file_name):
    with open(file_name, "r") as f:
        return [line.strip() for line in f]


def len_words(text):
    return len(text.split())


# rake z https://www.researchgate.net/publication/227988510_Automatic_Keyword_Extraction_from_Individual_Documents

class FeatureExtractor:
    def __init__(self):
        self.lemamatizer = WordNetLemmatizer()
        """ Features set containing unique words"""
        self.vocabulary = set()  # TODO return sorted
        self.phrases = set()

        self.stop_word_regex = self._create_stopwords_regex()

        self.max_words_in_feature = 3
        self.min_words_in_feature = 2
        self.min_keyword_frequency = 2
        self.min_word_length = 3

    def _create_stopwords_regex(self):  # todo usunac self
        self.stop_words = load_stop_words(STOP_LIST_FILE)
        words_to_remove_with_reg = [r"\b" + w + r"\b" for w in self.stop_words]
        words_to_remove_with_reg.append("\$?\d+[^\s]*")  # match number, $, %
        return re.compile('|'.join(words_to_remove_with_reg), re.IGNORECASE)

    def extract_features(self, tweet):
        def _extract_phrase(text, phrase):
            text, is_found = re.subn(phrase, '', text)
            return text, bool(is_found)

        features = {}
        sentences = preprocess(tweet)

        # TODO tutaj np sprawdzanie phrsaes czy spelniaja wymogi

        words = set()
        for s in sentences:
            for p in self.phrases:
                s, features[p] = _extract_phrase(s, p)
            # sentence has no feature phrases now
            chunks = self.split_by_stop_words(s)
            for c in chunks:
                lemmatized = self.lemamatize_many(c.split())
                words.update(lemmatized)

        for w in self.vocabulary:
            features[w] = (w in words)
        return features

    def build(self, dataset):
        all_tweets = [t for t, s in dataset]
        sentences = preprocess_many(all_tweets)
        candidates, words = self.generate_phrases(sentences)
        words = self.lemamatize_many(words)
        words = [w for w in words if len(w) >= self.min_word_length]
        return candidates, words

    def split_by_stop_words_many(self, texts):
        result = []
        for t in texts:
            result += self.split_by_stop_words(t)
        return result

    def split_by_stop_words(self, text):
        tmp = re.sub(self.stop_word_regex, '|', text.strip())
        phrases = tmp.split("|")
        phrases = [p.strip() for p in phrases]
        return [p for p in phrases if p]

    def is_phrase_length_ok(self, p):
        length_ok = self.min_words_in_feature <= len_words(p) <= self.max_words_in_feature
        return length_ok

    def lemamatize_many(self, words):
        return [self.lemamatizer.lemmatize(w) for w in words]

    def extract_not_matching_candidates(self, phrase):
        candidates, rest = set(), set()
        for p in phrase:
            if phrase.count(p) >= self.min_keyword_frequency and self.is_phrase_length_ok(p):
                candidates.add(p)
            else:
                rest.update(p.split())
        return candidates, rest

    def generate_phrases(self, tweets):
        phrases = []
        for t in tweets:
            for p in self.split_by_stop_words(t):
                p = clear_from_punct(p)
                if p:
                    phrases.append(p)

        candidates, rest = self.extract_not_matching_candidates(phrases)
        # candidates.update(self.extract_adjoined_candidates(tweets))
        return candidates, rest

    def build_vocabulary(self, dataset):
        self.phrases, all_words = self.build(dataset)
        self.vocabulary = set(all_words)
        print('VOC LENGTH' + str(len(self.vocabulary)))
        print('PHr LENGTH' + str(len(self.phrases)))

    # def extract_adjoined_candidates(self, tweets):  # only one that is found is "build the wall"
    #     candidates = []
    #     for t in tweets:
    #         words = t.split()
    #         words_len = len(words)
    #         if words_len > 3:
    #             for i in range(words_len - 2):
    #                 if not self.is_stopword(words[i]) \
    #                         and self.is_stopword(words[i+1]) \
    #                         and not self.is_stopword(words[i+2]):
    #                     candidates.append(" ".join(words[i:i+3]))
    #
    #     result = set()
    #     for c in candidates:
    #         if candidates.count(c) >= self.min_keyword_frequency:
    #             result.add(c)
    #     return result

    def is_stopword(self, w):
        return w in self.stop_words


def clear_from_punct(phrase):
    """ # todo usuwa tez se środka
    >>> clear_from_punct("Hello!")
    'Hello'
    >>> clear_from_punct(".")
    ''
    """
    return phrase.translate(punct_remove_translator)


def preprocess_many(tweets):
    """
    Lowers all text and splits into parts delimited by . or , or ? etc.
    >>> preprocess_many(["One. Two", "Next tweet: abc"])
    ['one', 'two', 'next tweet', 'abc']
    """
    result = []
    for t in tweets:
        result += preprocess(t)
    return result


def preprocess(tweet):
    """
    Lowers all text and splits into parts delimited by . or , or ? etc.
    >>> preprocess("One sentence. Another Part, and one more?")
    ['one sentence', 'another part', 'and one more']
    """

    def _split_sentences(text):
        sentence_delimiters = re.compile(u'[\\[\\]\n.!?,;:\t\\-\\"\\(\\)\\\'\u2019\u2013]')
        sentences = sentence_delimiters.split(text)
        sentences = [s.strip() for s in sentences]
        return sentences

    return _split_sentences(tweet.lower())


if __name__ == "__main__":
    import doctest

    doctest.testmod()
