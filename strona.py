from flask import Flask, render_template, request
from sentiment import get_tweet_sentiment

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():
    sentiment, polarity, = "", ""
    if request.method == 'POST':
        print("Teraz POST jest")
        print(request.form['tweet_content'])
        sentiment, pos, neg = get_tweet_sentiment(request.form['tweet_content'])
    return render_template('index.html', sentiment=sentiment, polarity=polarity)#=polarity)
