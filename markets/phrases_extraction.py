import re
import string
import os
from nltk.stem import WordNetLemmatizer
from sortedcontainers import SortedSet

STOP_LIST_FILE = os.path.join(os.path.dirname(__file__), "data/SmartStoplist.txt")
PUNCT_REMOVE_TRANSLATOR = str.maketrans('', '', string.punctuation)


def read_stop_words():
    with open(STOP_LIST_FILE, "r") as f:
        return [line.strip() for line in f]


def get_stopwords_regex():
    stop_words = read_stop_words()
    words_to_remove_with_reg = [r"\b" + w + r"\b" for w in stop_words]
    words_to_remove_with_reg.append(r"\$?\d+[^\s]*")  # match number, $, %
    return re.compile('|'.join(words_to_remove_with_reg), re.IGNORECASE)


# rake z https://www.researchgate.net/publication/227988510_Automatic_Keyword_Extraction_from_Individual_Documents

class PhrasesExtractor: # todo self greeedy?
    def __init__(self, min_keyword_frequency=2):
        """ Features set containing unique words"""
        self._vocabulary = SortedSet()
        self._phrases = SortedSet(key=phrases_sorting_key)
        self.stop_word_regex = get_stopwords_regex()

        self.max_words_in_feature = 3
        self.min_words_in_feature = 2
        self.min_keyword_frequency = min_keyword_frequency
        self.min_word_length = 3

    def set_words_and_phrases(self, words, phrases):
        self._vocabulary = SortedSet(words)
        self._phrases = SortedSet(phrases, key=phrases_sorting_key)

    def set_features(self, features):
        phrases = [f for f in features if ' ' in f]
        words = [f for f in features if f not in phrases]
        self.set_words_and_phrases(words, phrases)

    @property
    def features(self):
        return list(self._phrases) + list(self._vocabulary)

    def extract_features(self, tweet, greedy=True): # lepiej jest bez false
        features = dict.fromkeys(self._phrases, False)
        sentences = preprocess(tweet)

        extracted_words = set()
        for s in sentences:
            s, found_phrases = extract_phrases_from_text(s, self._phrases, greedy)
            for p in found_phrases:
                features[p] = True

            chunks = self.split_by_stop_words(s)
            for c in chunks:
                lemmatized = lemamatize_many(c.split())
                extracted_words.update(lemmatized)

        for w in self._vocabulary:
            features[w] = (w in extracted_words)
        return features

    def build_vocabulary(self, tweets):
        sentences = preprocess_many(tweets)
        phrases, words = self.generate_phrases(sentences)
        words = lemamatize_many(words)
        words = [w for w in words if len(w) >= self.min_word_length]
        return phrases, words

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

    def is_phrase_acceptable(self, candidate, candidates):
        has_occured_enough = candidates.count(candidate) >= self.min_keyword_frequency
        is_phrase_length_ok = self.min_words_in_feature <= len(candidate.split()) <= self.max_words_in_feature
        return has_occured_enough and is_phrase_length_ok

    def extract_not_matching_candidates(self, candidates):
        phrases = SortedSet(key=phrases_sorting_key)
        rest = set()

        for c in candidates:
            if self.is_phrase_acceptable(c, candidates):
                phrases.add(c)
            else:
                rest.add(c)

        # TODO czy to konieczne?
        rest_with_removed_phrases = SortedSet()  # remove phrases from non matching candidates
        for r in rest:
            r, _ = extract_phrases_from_text(r, phrases)
            rest_with_removed_phrases.update(r.split())
        return phrases, rest_with_removed_phrases

    def generate_phrases(self, tweets):
        phrases = []
        for t in tweets:
            for p in self.split_by_stop_words(t):
                p = clear_from_punct(p)
                if p:
                    phrases.append(p)

        candidates, rest = self.extract_not_matching_candidates(phrases)
        return candidates, rest

    def build(self, dataset):
        found_phrases, all_words = self.build_vocabulary(dataset)
        self.set_words_and_phrases(all_words, found_phrases)
        # print('VOCABULARY LEN ' + str(len(self._vocabulary)))
        # print('PHRASES LENGTH ' + str(len(self._phrases)))


def lemamatize_many(words):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in words]


def phrases_sorting_key(text):
    """
    >>> phrases_sorting_key("Big winner"), phrases_sorting_key("Big win")
    ((-2, -10), (-2, -7))
    """
    return -len(text.split()), -len(text)


def extract_phrases_from_text(text, phrases, greedy=True): # todo co jak na false?
    found = []
    for p in phrases:
        if greedy:
            text, is_found = re.subn(p, '', text)
            if is_found:
                found.append(p)
        else:
            matched = re.search(p, text)
            if matched:
                found.append(p)
    return text, found


def clear_from_punct(phrase):
    """ # todo usuwa tez se środka
    >>> clear_from_punct("Hello!")
    'Hello'
    >>> clear_from_punct(".")
    ''
    """
    return phrase.translate(PUNCT_REMOVE_TRANSLATOR)


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
    >>> preprocess("One #sentence. Another Part, and one mo-re? @DRUDGE_REPORT: aaa")
    ['one sentence', 'another part', 'and one more', 'drudge report', 'aaa']
    """

    sentence_delimiters = re.compile(u'[\[\]\n\.!\?,;\/:\t\"\(\)\u2019\u2013]|\s[\-]|[\-]\s')
    chars_to_remove = re.compile('[@#\-]')
    chars_to_replace = re.compile('[_]')

    tweet = tweet.lower()
    sentences = sentence_delimiters.split(tweet)
    sentences = [s.strip() for s in sentences]
    sentences = [re.sub(chars_to_remove, "", s) for s in sentences]
    sentences = [re.sub(chars_to_replace, " ", s) for s in sentences]
    return sentences


if __name__ == "__main__":
    import doctest

    doctest.testmod()
