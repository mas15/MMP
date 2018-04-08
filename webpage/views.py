from flask import render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import DataRequired, Length
from wtforms.widgets import TextArea
from webpage import app


class TweetTextForm(FlaskForm):
    tweet_content = StringField('tweet_content', validators=[DataRequired(), Length(3, 300)], widget=TextArea())


CURRENCIES = {"dollar": "USD", "euro": "EUR", "mexico": "MEX"}


@app.route('/', methods=['POST', 'GET'], defaults={'currency': 'dollar'})
@app.route('/currency/<currency>', methods=['POST', 'GET'])
def index(currency):
    form = TweetTextForm()  # todo validate currency
    currency_short = CURRENCIES[currency]
    analyser = app.analysers[currency_short]

    prediction_results = dict()

    if form.validate_on_submit():
        prediction_results = analyser.analyse_tweet(request.form["tweet_content"])

    return render_template('currency.html',
                           prediction=prediction_results,
                           currency=currency,
                           form=form,
                           graph_data=analyser.get_graph_data(),
                           features_data=analyser.get_most_coefficient_features(),
                           rules_data=analyser.get_rules_data())
