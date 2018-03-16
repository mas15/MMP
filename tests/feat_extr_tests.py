from markets.feature_extractor import FeatureExtractor, extract_phrases_from_text, preprocess
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

    def test_generate_phrases(self):
        sentences = ["fake news media", "fake news media again", "bad fake news media"]
        phrases, words = self.extr.generate_phrases(sentences)
        self.assertEqual(["fake news media"], list(phrases))
        self.assertEqual({"bad"}, words)

    def test_extract_phrases_from_text(self):  # todo przeniesc
        phrases = ["fake news media", "hillary clinton"]
        text = "bad fake news media"
        res_words, res_phrases = extract_phrases_from_text(text, phrases)
        self.assertEqual("bad ", res_words)
        self.assertEqual(["fake news media"], res_phrases)

    def test_extract_phrases_from_text_2(self):  # todo przeniesc
        phrases = ["tax cuts", "fake news media", "hillary clinton"]
        text = "bad fake news media about hillary clinton and more text"
        res_words, res_phrases = extract_phrases_from_text(text, phrases)
        self.assertEqual("bad  about  and more text", res_words)
        self.assertEqual(["fake news media", "hillary clinton"], res_phrases)

    def test_extract_features_2(self):
        self.extr.vocabulary = ["hello", "medium", "feature"]
        self.extr.phrases = ["whole phrases here", "make america great", "tax cuts", "hilary clinton"]

        res = self.extr.extract_features("Media talking about tax cuts")
        exp_res = {k: False for k in self.extr.vocabulary + self.extr.phrases}
        exp_res["tax cuts"] = True
        exp_res["medium"] = True
        self.assertEqual(res, exp_res)

    dataset = ["one sentence there",
               "Make America great again. fake news media.",
               "another tweet about Hillary Clinton. Fake news media one more time. ",
               "crooked Hillary Clinton again, wall with Mexico, next tax cuts",
               "Very proud of my Executive Order which will allow greatly expanded "
               "access and far lower costs for HealthCare. Fake news media and Hillary Clinton again",
               ]
    exp_features = [
        ["sentence"],
        ['america', 'fake news media', 'great', 'make'],
        ['fake news media', 'hillary clinton', 'time', 'tweet'],
        ['crooked', 'cut', 'hillary clinton', 'mexico', 'tax', 'wall'],
        ['access', 'cost', 'executive', 'expanded', 'fake news media', 'greatly', 'healthcare',
         'hillary clinton', 'lower', 'order', 'proud']
    ]

    def test_build_vocabulary(self):
        self.extr.build_vocabulary(self.dataset)
        print(self.extr.phrases)
        print(self.extr.vocabulary)

        exp_phrases = ['fake news media', 'hillary clinton']
        exp_vocabulary = set([word for words in self.exp_features for word in words]) - set(exp_phrases)
        self.assertEqual(sorted(exp_vocabulary), sorted(self.extr.vocabulary))
        self.assertEqual(exp_phrases, sorted(self.extr.phrases))

    def test_build_vocabulary_and_extract(self):
        self.extr.build_vocabulary(self.dataset)
        print(self.extr.phrases)
        print(self.extr.vocabulary)
        for t, exp_res in zip(self.dataset, self.exp_features):
            extracted = self.extr.extract_features(t)
            found = [f for f, is_found in extracted.items() if is_found]
            self.assertEqual(exp_res, sorted(found))  # todo usunac sorted ale dac ordered set

    def test_split_by_stop_words(self):
        tweet = "Just cannot  believe a judge would put our country in such peril " \
                "If something happens blame him and court system or People pouring in Bad "
        res = self.extr.split_by_stop_words(tweet)
        exp_res = ['judge', 'put', 'country', 'peril', 'blame', 'court system', 'People pouring', 'Bad']
        self.assertEqual(res, exp_res)

    def test_is_phrase_acceptable(self):
        self.extr.min_keyword_frequency = 0  # todo test czy acc jak wystarczajaco razy
        self.assertFalse(self.extr.is_phrase_acceptable("Four is too much", []))
        self.assertTrue(self.extr.is_phrase_acceptable("Three words ok", []))
        self.assertTrue(self.extr.is_phrase_acceptable("Also ok", []))
        self.assertFalse(self.extr.is_phrase_acceptable("Bad", []))

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

        exp_phrases = ['Make america great', 'Hillary Clinton', 'lowest level']
        exp_rest = ['base', 'jail', 'prosecuted', 'stronger', 'unemployment', 'years']

        phrases, rest = self.extr.generate_phrases(tweets)
        self.assertEqual(exp_phrases, list(phrases))
        self.assertEqual(exp_rest, sorted(rest))

    def test_extracting_phrases_with_apostrophe(self):
        phrases, rest = self.extr.generate_phrases(["who is obviously a madman who doesn't mind starving"])
        self.assertEqual([], list(phrases))
        self.assertEqual(['madman', 'mind', 'starving'], sorted(rest))

    def test_splitting_sentences(self):
        sent = "Business is looking better than ever with business enthusiasm at record levels. " \
            "Stock Market at an all-time high. That doesn't just happen! Mexico, wall"
        result = preprocess(sent)
        exp_result = ['business is looking better than ever with business enthusiasm at record levels',
                      'stock market at an all-time high', "that doesn't just happen", 'mexico', 'wall']
        self.assertEqual(exp_result, result)


if __name__ == '__main__':
    unittest.main()
