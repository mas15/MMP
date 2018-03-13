import csv
from datetime import datetime
from markets.sentiment import SentimentAnalyser
from markets.feature_extractor import FeatureExtractor

def read_stock():
    result = []
    with open('USDIndex.csv', 'r', encoding='utf8') as f:
        reader = csv.reader(f, delimiter=",")
        next(reader, None)  # skip the headers
        try:
            for line in reader:
                date, price, open_p, high, low, vol, change = line
                x = datetime.strptime(date, '%b %d, %Y'), price
                result.append(x)
        except IndexError:
            pass
    return result


def read_tweets():
    res =[]
    with open('all_tweets.csv', 'r', encoding='utf8') as f:
        reader = csv.reader(f, delimiter=",")
        try:
            for line in reader:
                id, content, created_at = line
                date = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
                res.append((content, date))
        except IndexError:
            pass
        return res



tweets = read_tweets()
# tweets = [t for t, d in tweets_with_dates] # todo use dates
ex = FeatureExtractor()
ex.build_vocabulary(tweets) # todo uzywac tylko tweets bez sent/dates
print("VOCABULARY:")
print(len(ex.vocabulary))
# print(ex.vocabulary)
print(len(ex.phrases))
# print(ex.phrases)


sent = SentimentAnalyser()
sent.load()

from collections import OrderedDict

with open('temp.csv', 'w', encoding='utf-8') as fw:
        writer = csv.writer(fw)
        first = True
        for t, _ in tweets:
            features = ex.extract_features(t)
            features = OrderedDict(sorted(features.items()))

            sentiment = sent.analyse(t)
            line = [sentiment] + [v for v in features.values()]

            if first:
                writer.writerow(["sentiment"] + [k for k in features.keys()])
            writer.writerow(line)
            first = False