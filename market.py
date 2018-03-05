import csv
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib

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
    with open('aaa.csv', 'r', encoding='utf8') as f:
        reader = csv.reader(f, delimiter=",")
        try:
            for line in reader:
                date, content, sentiment = line[2], line[1], line[3]
                date = datetime.strptime(date.split()[0], '%Y-%m-%d')
                res.append((content, sentiment, date))
        except IndexError:
            pass
        return res



for t in read_tweets():
    _, s, d = t
    for s in read_stock():
        if s[0] == d:
