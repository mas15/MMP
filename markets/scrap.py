# -*- coding: utf-8 -*-
import tweepy
from datetime import datetime
import csv


class TweeterScrapper:
    def __init__(self):
        consumer_key = "sHEmHwtt3koxdLoa6Ok2vEduH"
        consumer_secret = "fJZsN0OQW80Vqnw265rT8Jvc7VwADGNS0kB5vMjIRG4d3eywzJ"
        access_token = "962374758783471616-KZtDMvJkmJigxZWUr3EI8x5iOgguRQB"
        access_token_secret = "hLtD8RMXyT2kcUDR9oLg7P7MtXSkrlgBWgsZk4u8GJY84"

        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)

        self.api = tweepy.API(auth)

    def get_tweets(self, user, count):  # cant get more than 200 at once - count=200,max_id=201)
        return self.api.user_timeline(user, count=count, tweet_mode="extended", )

    def get_status(self, id):
        return self.api.get_status(id)

# jak CSV: https://gist.github.com/yanofsky/5436496

    def get_all_since(self, date):
        def store(tweets, writer):
            for t in tweets:
                print(t.full_text)
                print()
                fields = [t.id_str, t.full_text, t.created_at]
                writer.writerow(fields)

        with open('all_tweets.csv', 'a', encoding='utf-8') as f:
            writer = csv.writer(f)
            last_id = 913004378486984704# 970650759091163137
            last_date = datetime.today()
            while last_date > date:
                # while len(tweets) > 0:
                tweets = scrapper.api.user_timeline("realDonaldTrump", count=200, tweet_mode="extended", max_id=last_id)
                store(tweets[1:], writer)
                last_id = tweets[199].id
                last_date = tweets[199].created_at


if __name__ == "__main__":
    # scrapper = TweeterScrapper()
    # scrapper.get_all_since(datetime(2017,1,1,0,0,0))
    #
    # for id in [816260343391514624]:
    #     t = scrapper.get_status(id)
    #     print(t.full_text)
    #
    #     fields = [t.id_str, t.full_text, t.created_at, 'neg']
    #     with open('sentimental_tweets.csv', 'a', encoding='utf8') as f:
    #         writer = csv.writer(f, newline='')
    #         writer.writerow(fields)

    import re
    i=0
    with open('temp.csv', 'w', encoding='utf-8', newline='') as fw:
        writer = csv.writer(fw)
        with open('all_tweets.csv', 'r', encoding='utf8') as fr:
            reader = csv.reader(fr, delimiter=",")
            for line in reader:
                i+=1
                try:
                    #line[1] = re.sub(r'\S?https\S+', '', line[1])
                    #line[1] = line[1].replace("\n", "")
                    writer.writerow(line)
                except IndexError:
                    pass