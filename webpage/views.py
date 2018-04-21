from flask import render_template, request, abort
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
    form = TweetTextForm()
    prediction_results = dict()
    try:
        currency_details = Currency.get_currency(currency)  # todo raise exception
        analyser = app.analysers[currency]

        if form.validate_on_submit():
            prediction_results = analyser.analyse_tweet(request.form["tweet_content"])  # todo co jak exception?

        return render_template('currency.html',
                               currencies=Currency.get_all(),
                               prediction=prediction_results,
                               currency_details=currency_details,
                               form=form,
                               graph_data=analyser.get_graph_data(),
                               features_data=analyser.get_most_coefficient_features(),
                               rules_data=analyser.get_rules_data())
    except KeyError:
        abort(404)

