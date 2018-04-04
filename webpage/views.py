from flask import render_template, request
from markets.association import get_graph_data
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
        prediction_results = app.model.analyse(request.form["tweet_content"])

    graph_data = get_graph_data()
    features_data = app.model.get_most_coefficient_features()

    return render_template('currency.html', prediction=prediction_results, currency=currency,
                           form=form, graph_data=graph_data, features_data=features_data)



