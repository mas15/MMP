from flask import render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import DataRequired, Length
from wtforms.widgets import TextArea
from webpage import app
from webpage.models import Currency


class TweetTextForm(FlaskForm):
    tweet_content = StringField('tweet_content', validators=[DataRequired(), Length(3, 300)], widget=TextArea())


@app.route('/', methods=['POST', 'GET'], defaults={'currency': 'USD'})
@app.route('/currency/<currency>', methods=['POST', 'GET'])
def index(currency):
    form = TweetTextForm()  # todo validate currency
    analyser = app.analysers[currency]
    currency_details = Currency.get_currency(currency)

    prediction_results = dict()

    if form.validate_on_submit():
        prediction_results = analyser.analyse_tweet(request.form["tweet_content"])

    return render_template('currency.html',
                           prediction=prediction_results,
                           currency_details=currency_details,
                           form=form,
                           graph_data=analyser.get_graph_data(),
                           features_data=analyser.get_most_coefficient_features(),
                           rules_data=analyser.get_rules_data())
