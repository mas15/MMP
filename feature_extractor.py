from rake import Rake, load_stop_words
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import re


# rake z https://www.researchgate.net/publication/227988510_Automatic_Keyword_Extraction_from_Individual_Documents

class FeatureExtractor:
    def __init__(self):
        self.lemamatizer = WordNetLemmatizer()
        self.r = Rake("SmartStoplist.txt", 3, 3, 2)  # todo
        """ Features set containing unique words"""
        self.vocabulary = set()
        self.phrases = []

        self.stop_word_regex = self._create_stopwords_regex()

        self.max_words_in_feature = 3
        self.min_words_in_feature = 2
        self.min_keyword_frequency = 2
        self.min_word_length = 3

    def _create_stopwords_regex(self):  # todo usunac self
        self.words_to_remove = load_stop_words("SmartStoplist.txt") + ["n't", "'s", "\.\.\."]  # shouldn't be split
        words_to_remove_with_reg = [r"\b" + w + r"\b" for w in self.words_to_remove]
        words_to_remove_with_reg += r"[!\"#$%&'()\*\+,\-\.:;<=>?@\^_`{|}~’]"  # todo jak to ogarnac? i to wyzej tez
        return re.compile('|'.join(words_to_remove_with_reg), re.IGNORECASE)

    def extract_features(self, tweet):  # todo to zostaje bo nie bylo wgl rake tutaj
        features = {}
        for phrase in self.phrases:
            features[phrase] = (phrase in tweet.lower())
            tweet = tweet.replace(phrase, "")  # TODO not sure if not spoil tokenizing

        words_in_tweet = self.extract_words_from_tweet(tweet)
        for word in self.vocabulary:
            features[word] = (word in words_in_tweet)
        return features

    def extract_words_from_tweet(self, tweet):
        # todo lower sreipl, current_word != '' and not is_number(current_word): nowe linie
        def _to_skip(w):
            return w in self.words_to_remove or w.replace(".", "").isdigit()

        def _clear(w):
            return w.replace("`", "").replace("'", "").replace("\"", "").replace("\n", "").replace("\\", "")

        words = word_tokenize(tweet)  # albo split
        words = [w.lower() for w in words]
        words = [w for w in words if not _to_skip(w)]
        words = [_clear(w) for w in words]
        words = [self.lemamatizer.lemmatize(w) for w in words if w]  # tu spradza czy cos zostalo
        return words

    def build(self, dataset):
        tweets = []
        for t, s in dataset:
            tweets += self.split_sentences(t.lower())
        candidates = self.generate_candidate_keywords(tweets)
        #words_score = self.calculate_word_scores(candidates)
        #candidates_scores = self.generate_candidate_keyword_scores(candidates, words_score)
        #phrases = sorted(candidates_scores.items(), key=lambda x: x[1])

        phrases, words = [], []
        for c in candidates:
            if phrases.count(c) >= self.min_keyword_frequency:
                phrases.append(c)
            else:
                words.extend(c.split())
        return phrases, words


    # def generate_candidate_keyword_scores(self, phrases, words_scores):
    #     keyword_candidates = defaultdict(int)
    #     for phrase in phrases:
    #         if phrases.count(phrase) >= self.min_keyword_frequency:  # todo czemu dopiero tutaj?
    #             phase_score = [words_scores[word] for word in
    #                            phrase.split()]  # todo co inaczej niz extract from tweets?
    #             keyword_candidates[phrase] = phase_score
    #     return keyword_candidates

    def calculate_word_scores(self, phrases):
        word_frequency = defaultdict(int)
        word_degree = defaultdict(int)
        for p in phrases:
            words = p.split()
            words_degree = len(words) - 1  # todo usunac 1 i z dolu + feq
            # if word_list_degree > 3: word_list_degree = 3 #exp.
            for word in words:
                word_frequency[word] += 1
                word_degree[word] += words_degree  # orig.
                # word_degree[word] += 1/(word_list_length*1.0) #exp.

        # Calculate Word scores = deg(w)/freq(w) # todo zmienilem na deg(w)+freq(w) bo i tak dodawlo do freq/freq(w)
        words_score = defaultdict(int)
        for word in word_frequency:
            words_score[word] = word_degree[word] + word_frequency[word] / (word_frequency[word] * 1.0)  # orig.
            # word_score[item] = word_frequency[item]/(word_degree[item] * 1.0) #exp.
        return words_score

    def split_by_stop_words(self, text):
        tmp = re.sub(self.stop_word_regex, '|', text.strip())
        phrases = tmp.split("|")
        return [p.strip() for p in phrases]

    def is_acceptable(self, p):
        words = p.split()
        length_ok = self.min_words_in_feature <= len(words) <= self.max_words_in_feature
        # each_word_length = czy kazde > 3 # todo też z cyframi co jesli?
        return length_ok

    def generate_candidate_keywords(self, tweets):
        phrases = []
        for t in tweets:
            phrases += self.split_by_stop_words(t)
        phrases = [p for p in phrases if self.is_acceptable(p)]

        # TODO phrases += extract_adjoined_candidates(tweets)
        return phrases

    def split_sentences(self, text):
        """
        Utility function to return a list of sentences.
        @param text The text that must be split in to sentences.
        """
        sentence_delimiters = re.compile(u'[\\[\\]\n.!?,;:\t\\-\\"\\(\\)\\\'\u2019\u2013]')
        sentences = sentence_delimiters.split(text)
        sentences = [s.strip() for s in sentences]
        return sentences

    def build_vocabulary(self, dataset):
        # all_words = []
        # phrases = self.build(dataset)  # todo czy one juz lower sa?
        #
        # self.phrases = [phrase for phrase, score in phrases if " " in phrase]
        #
        # for tweet, s in dataset:
        #     all_words += self.extract_words_from_tweet(tweet)
        # self.vocabulary = list(set(all_words))

        self.phrases, all_words = self.build(dataset)
        self.vocabulary = list(set(all_words))
