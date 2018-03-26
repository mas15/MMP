from flask import render_template, request
from markets.association import get_date_to_check_affect
from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import DataRequired, Length
from wtforms.widgets import TextArea
from webpage import app


class TweetTextForm(FlaskForm):
    tweet_content = StringField('tweet_content', validators=[DataRequired(), Length(3, 300)], widget=TextArea())


@app.route('/', methods=['POST', 'GET'], defaults={'currency': 'dollar'})
@app.route('/currency/<currency>', methods=['POST', 'GET'])
def index(currency):
    form = TweetTextForm() # todo validate currency
    prediction_results = dict()
    if form.validate_on_submit():
        print(request.form["tweet_content"])
        prediction_results = app.model.predict(request.form["tweet_content"])
    graph_data = get_graph_data()
    features_data = get_features_data()
    return render_template('currency.html', prediction=prediction_results, currency=currency,
                           form=form, graph_data=graph_data, features_data=features_data)


def get_features_data():
    return app.model.get_most_coefficient_features()


def get_graph_data():
    from markets.association import read_all_tweets, read_dollar_prices

    all_tweets = read_all_tweets()
    dollar_prices = read_dollar_prices()
    all_tweets.drop(columns=["Id"], inplace=True)
    dollar_prices.drop(columns=["Price", "High", "Low", "Market_change"], inplace=True)
    all_tweets["Date"] = all_tweets["Date"].apply(get_date_to_check_affect)
    all_tweets["Date"] = all_tweets["Date"].dt.strftime('%Y-%m-%d')
    dollar_prices["Date"] = dollar_prices["Date"].dt.strftime('%Y-%m-%d')

    tweets_per_date = dict(zip(all_tweets.Date, all_tweets.Text))
    labels = dollar_prices["Date"].values.tolist()
    vals = dollar_prices["Open"].values.tolist()

    return labels, vals, tweets_per_date
