from flask import Flask, render_template, request, flash
from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import DataRequired, Length
from wtforms.widgets import TextArea
from sentiment import SentimentAnalyser
from mmm import get_date_to_check_affect

app = Flask(__name__, static_url_path='/static')
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['DEBUG'] = True
app.config['SECRET_KEY'] = b'\xdfP\xdb\xc9\xe4K\x0fc\x10\x06\xca\xaf\x1f\xb3\x00x\xc6\xd2\x96 lg\xf7\xad'

app.sent = SentimentAnalyser()


class TweetTextForm(FlaskForm):
    tweet_content = StringField('tweet_content', validators=[DataRequired(), Length(3, 300)], widget=TextArea())


@app.route('/', methods=['POST', 'GET'])
@app.route('/index', methods=['POST', 'GET'])
@app.route('/dollar', methods=['POST', 'GET'])
def dollar_page():
    form = TweetTextForm()
    sentiment = ""
    if form.validate_on_submit():
        print(request.form["tweet_content"])
        sentiment = app.sent.analyse(request.form["tweet_content"])
    labels, values, tweets = get_graph_data()
    return render_template('currency.html', sentiment=sentiment, currency="dollar",
                           form=form, labels=labels, values=values, tweets=tweets)


@app.route('/euro', methods=['POST', 'GET'])
def euro_page():
    form = TweetTextForm()
    sentiment = ""
    if form.validate_on_submit():
        print(request.form["tweet_content"])
        sentiment = app.sent.analyse(request.form["tweet_content"])
    labels, values, tweets = get_graph_data()
    return render_template('currency.html', sentiment=sentiment, currency="euro",
                           form=form, labels=labels, values=values, tweets=tweets)


def get_graph_data():
    from mmm import read_all_tweets, read_dollar_prices

    all_tweets = read_all_tweets()
    dollar_prices = read_dollar_prices()
    all_tweets.drop(columns=["Id"], inplace=True)
    dollar_prices.drop(columns=["Price", "High", "Low", "Change"], inplace=True)
    all_tweets["Date"] = all_tweets["Date"].apply(get_date_to_check_affect)
    all_tweets["Date"] = all_tweets["Date"].dt.strftime('%Y-%m-%d')
    dollar_prices["Date"] = dollar_prices["Date"].dt.strftime('%Y-%m-%d')

    tweets_per_date = dict(zip(all_tweets.Date, all_tweets.Text))
    labels = [x for x in dollar_prices["Date"].values]
    vals = [x for x in dollar_prices["Open"].values]

    return labels, vals, tweets_per_date


if __name__ == "__main__":
    app.run()
