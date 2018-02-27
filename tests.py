from feature_extractor import FeatureExtractor
import unittest


class TestFeatureExtractor(unittest.TestCase):
    def test_extract_features(self):
        extr = FeatureExtractor()
        extr.vocabulary = ["hello", "another", "feature", "great"]  # todo again?
        extr.phrases = ["whole phrases here", "make america great", "tax cuts", "hilary clinton"]

        res = extr.extract_features("Make America great again")
        exp_res = {k: False for k in extr.vocabulary + extr.phrases}
        exp_res["make america great"] = True
        exp_res["great"] = True  # CZY TO DOBRZE?
        self.assertEqual(res, exp_res)

    def test_extract_features_2(self):
        extr = FeatureExtractor()
        extr.vocabulary = ["hello", "medium", "feature"]
        extr.phrases = ["whole phrases here", "make america great", "tax cuts", "hilary clinton"]

        res = extr.extract_features("Media talking about tax cuts")
        exp_res = {k: False for k in extr.vocabulary + extr.phrases}
        exp_res["tax cuts"] = True
        exp_res["medium"] = True
        self.assertEqual(res, exp_res)

    def test_build_vocabulary(self):
        extr = FeatureExtractor()
        dataset = [("one sentence there", "pos"),
                   ("Make America great again. fake news media.", "pos"),
                   ("another tweet about Hillary Clinton. Fake news media one more time. ", "pos"),
                   ("crooked Hillary Clinton again, wall with Mexico, next tax cuts", "pos"),
                   ("Very proud of my Executive Order which will allow greatly expanded "
                    "access and far lower costs for HealthCare. Fake news media and Hillary Clinton again", "pos")
                   ]
        extr.build_vocabulary(dataset) # TODO czy clinton powinna byc w words?
        exp_vocabulary = ['access', 'america', 'clinton', 'cost', 'crooked', 'cut',
                          'executive', 'expanded', 'fake', 'great', 'greatly', 'healthcare',
                          'hillary', 'lower', 'make', 'medium', 'mexico', 'news', 'order',
                          'proud', 'sentence', 'tax', 'time', 'tweet', 'wall']
        exp_phrases = ['fake news media', 'hillary clinton']
        self.assertEqual(sorted(extr.vocabulary), exp_vocabulary)
        self.assertEqual(sorted(extr.phrases), exp_phrases)

    def test_extract_words_from_tweet(self):
        extr = FeatureExtractor()
        tweet = "Just cannot  believe a judge would put our country in such peril. " \
                "If something happens blame him and court system. People pouring in. Bad! " \
                "Some words to lemmatize: taxes, cuts, walls"
        res = extr.extract_words_from_tweet(tweet)
        exp_res = ['judge', 'put', 'country', 'peril', 'blame', 'court', 'system', 'people', 'pouring',
                   'bad', 'word', 'lemmatize', 'tax', 'cut', 'wall']
        self.assertEqual(res, exp_res)

    def test_is_phrase_acceptable(self):
        extr = FeatureExtractor()
        self.assertFalse(extr.is_acceptable("Four is too much"))
        self.assertTrue(extr.is_acceptable("Three words ok"))
        self.assertTrue(extr.is_acceptable("Also ok"))
        self.assertFalse(extr.is_acceptable("Bad"))

    def test_split_by_stop_words(self):
        extr = FeatureExtractor()
        text = "Taxes and people are bad decisions. stupid people: blame"
        exp_res = ["Taxes", "people", "bad decisions", "stupid people", "blame"]
        res = extr.split_by_stop_words(text)
        self.assertEqual(res, exp_res)


if __name__ == '__main__':
    unittest.main()
