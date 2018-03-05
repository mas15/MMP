from feature_extractor import FeatureExtractor
from sentiment import SentimentAnalyser
import unittest


class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.extr = FeatureExtractor()

    def test_extract_features(self):
        self.extr.vocabulary = ["hello", "another", "feature", "great"]  # todo again?
        self.extr.phrases = ["whole phrases here", "make america great", "tax cuts", "hilary clinton"]

        res = self.extr.extract_features("Make America great again")
        exp_res = {k: False for k in self.extr.vocabulary + self.extr.phrases}
        exp_res["make america great"] = True
        # exp_res["great"] = True  # CZY TO DOBRZE?
        self.assertEqual(res, exp_res)

    def test_extract_features_2(self):
        self.extr.vocabulary = ["hello", "medium", "feature"]
        self.extr.phrases = ["whole phrases here", "make america great", "tax cuts", "hilary clinton"]

        res = self.extr.extract_features("Media talking about tax cuts")
        exp_res = {k: False for k in self.extr.vocabulary + self.extr.phrases}
        exp_res["tax cuts"] = True
        exp_res["medium"] = True
        self.assertEqual(res, exp_res)

    def test_build_vocabulary(self):
        dataset = [("one sentence there", "pos"),
                   ("Make America great again. fake news media.", "pos"),
                   ("another tweet about Hillary Clinton. Fake news media one more time. ", "pos"),
                   ("crooked Hillary Clinton again, wall with Mexico, next tax cuts", "pos"),
                   ("Very proud of my Executive Order which will allow greatly expanded "
                    "access and far lower costs for HealthCare. Fake news media and Hillary Clinton again", "pos")
                   ]
        self.extr.build_vocabulary(dataset)  # TODO czy clinton powinna byc w words?
        exp_vocabulary = ['access', 'america', 'clinton', 'cost', 'crooked', 'cut',
                          'executive', 'expanded', 'great', 'greatly', 'healthcare',
                          'hillary', 'lower', 'make', 'mexico', 'order',
                          'proud', 'sentence', 'tax', 'time', 'tweet', 'wall']
        exp_phrases = ['fake news media', 'hillary clinton']
        self.assertEqual(sorted(self.extr.vocabulary), exp_vocabulary)
        self.assertEqual(sorted(self.extr.phrases), exp_phrases)

    def test_split_by_stop_words(self):
        tweet = "Just cannot  believe a judge would put our country in such peril " \
                "If something happens blame him and court system or People pouring in Bad "
        res = self.extr.split_by_stop_words(tweet)
        exp_res = ['judge', 'put', 'country', 'peril', 'blame', 'court system', 'People pouring', 'Bad']
        self.assertEqual(res, exp_res)

    def test_is_phrase_acceptable(self):
        self.assertFalse(self.extr.is_phrase_length_ok("Four is too much"))
        self.assertTrue(self.extr.is_phrase_length_ok("Three words ok"))
        self.assertTrue(self.extr.is_phrase_length_ok("Also ok"))
        self.assertFalse(self.extr.is_phrase_length_ok("Bad"))

    def test_split_by_stop_words_2(self):
        text = "Taxes and people are bad decisions then stupid people $1.2 blame 45% and nothing 1st. time"
        exp_res = ["Taxes", "people", "bad decisions", "stupid people", "blame", "time"]
        res = self.extr.split_by_stop_words(text)
        self.assertEqual(res, exp_res)

    def test_generate_candidate_keywords(self):
        tweets = ["Make america great again.",
                  "unemployment at lowest level in years and our base has never been stronger!",
                  "Hillary Clinton should have been prosecuted and should be in jail."]
        self.extr.min_keyword_frequency = 1

        exp_cands = set(['Make america great', 'lowest level', 'Hillary Clinton'])
        exp_rest = ['unemployment', 'years', 'base', 'stronger', 'prosecuted', 'jail']

        cands, rest = self.extr.generate_candidate_keywords(tweets)
        self.assertEqual(cands, exp_cands)
        self.assertEqual(rest, exp_rest)


if __name__ == '__main__':
    unittest.main()
