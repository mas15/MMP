# import pandas as pd
# from markets.helpers import remove_features, move_column_to_the_end, mark_features, \
#     drop_instances_without_features
# from markets.sentiment import SentimentAnalyser, calculate_sentiment
# from markets.phrases_extractor import PhrasesExtractor
#
#
# class TweetFeaturesExtractor:
#     def __init__(self, features=None, extr=None, sent=None):
#         self.extr = extr or PhrasesExtractor(min_keyword_frequency=4)
#         if features:
#             self.extr.set_features(features)
#         self.sent = sent or SentimentAnalyser()
#         self.sent.load()
#
#     def extract_features(self, df):  # get features vector? # todo Feature Extractor/ Phrases extracot
#         if not self.extr.features:
#             self.extr.build_vocabulary(df["Text"].tolist())
#
#         df = mark_features(self.extr, df)
#         df = calculate_sentiment(df, self.sent)
#
#         if "Market_change" in list(df):
#             df = move_column_to_the_end(df, "Market_change")
#         return df
#
#     def filter_features(self, df, features):
#         features_to_leave = features + ["Tweet_sentiment", "Market_change", "Text"]
#         sifted_df = remove_features(df, features_to_leave)
#
#         self.extr.set_features(features)
#         sifted_df = mark_features(self.extr, sifted_df)
#         sifted_df = drop_instances_without_features(sifted_df)
#         return sifted_df
#
#     def process_text(self, text):
#         df = pd.DataFrame({'Text': [text]})
#         df = self.extract_features(df)
#         df.drop(columns=["Text"], inplace=True)
#         return df
