import tweepy


class TweeterScrapper:
    def __init__(self):
        consumer_key = "sHEmHwtt3koxdLoa6Ok2vEduH"
        consumer_secret = "fJZsN0OQW80Vqnw265rT8Jvc7VwADGNS0kB5vMjIRG4d3eywzJ"
        access_token = "962374758783471616-KZtDMvJkmJigxZWUr3EI8x5iOgguRQB"
        access_token_secret = "hLtD8RMXyT2kcUDR9oLg7P7MtXSkrlgBWgsZk4u8GJY84"

        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)

        self.api = tweepy.API(auth)

    def get_tweets(self, user, count): # cant get more than 200 at once - count=200,max_id=201)

        #user_acc = self.api.get_user(user)
        #return user_acc.timeline(count=count)
        return self.api.user_timeline(user, count=count)

    def get_status(self, id):
        return self.api.get_status(id)

# jak CSV: https://gist.github.com/yanofsky/5436496


if __name__ == "__main__":
    scrapper = TweeterScrapper()
    #tweets = scrapper.get_tweets("realDonaldTrump", 20)
    #for tweet in tweets:
    #    print(tweet.text.encode("utf-8"))
    #    print()

    # poss:

    import csv
    for id in [604411222906265600]:
        t = scrapper.get_status(id)
        print(t.text)

        fields = [t.id_str, t.text.encode("utf8"), t.created_at, 'neg']
        with open('aaa.csv', 'a', encoding='utf8') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

