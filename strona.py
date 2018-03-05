from flask import Flask, render_template, request
from sentiment import SentimentAnalyser

app = Flask(__name__)
app.sent = SentimentAnalyser() # todo czy to dobrze?


@app.route('/', methods=['POST', 'GET'])
def index():
    sentiment, polarity, = "", ""
    if request.method == 'POST':
        print("Teraz POST jest")
        print(request.form['tweet_content'])
        sentiment = app.sent.analyse(request.form['tweet_content'])
    return render_template('index.html', sentiment=sentiment, polarity=polarity)#=polarity)
