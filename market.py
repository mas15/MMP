import csv
from datetime import datetime
from sentiment import SentimentAnalyser
from feature_extractor import FeatureExtractor

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
ex = FeatureExtractor()
ex.build_vocabulary(tweets)
print("VOCABULARY:")
print(len(ex.vocabulary))
[print(w) for w in ex.vocabulary]
print(len(ex.phrases))
print(ex.phrases)
