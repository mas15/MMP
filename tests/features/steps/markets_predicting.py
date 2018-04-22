import os
import sys

sys.path.insert(0, os.path.pardir)

from behave import given, when, then, step
from markets.currency_analyser import CurrencyAnalyser
from markets.market_predicting_model import MarketPredictingModel
from unittest import mock
from sklearn.naive_bayes import MultinomialNB
import numpy as np


@given('we created a currency analyser')
def step_impl(context):
    def predict_proba(x):
        """
        Mocked out Scikit learns predict proba which normally returns array of arrays
        with predictions for each class. We mock out the result based on words found
        """
        if x[0][0]:  # Mexico
            result = [0.8, 0.5, 0.2]
        elif x[0][1]:  # Apples
            result = [0.3, 0.5, 0.4]
        elif x[0][2]:  # Good
            result = [0.1, 0.2, 0.9]
        else:  # no features found
            result = [0.2, 0.2, 0.2]

        if len(x) == 5:  # if all features model used
            if x[0][3] and x[0][4]:  # pizza and banana found
                result = [0.2, 0.1, 1]
            else:  # no features found
                result = [0.4, 0.4, 0.4]

        return [np.array(result)]

    mock_model = mock.create_autospec(MultinomialNB)
    mock_model.classes_ = ["Down", "NC", "Up"]
    mock_model.predict_proba = predict_proba
    with mock.patch("markets.market_predicting_model.MultinomialNB", return_value=mock_model):
        context.a = CurrencyAnalyser("abc")
        context.a._model = MarketPredictingModel()


@given('all the features are: {features}')
def step_impl(context, features):
    context.a._model.all_features = features.split(", ")


@given('the main features are: {features}')
def step_impl(context, features):
    context.a._model.main_features = features.split(", ")


#
# @given('we know their sentiment: {sentiments}')
# def step_impl(context, sentiments):
#     context.texts = list(zip(context.texts, sentiments.split(", ")))

#
# @when('we train a model')
# def step_impl(context):
#     context.s.train(context.texts)


@when('we analyse a tweet: {text}')
def step_impl(context, text):
    context.result = context.a.analyse_tweet(text)


@then('the result is {expected_result}')
def step_impl(context, expected_result):
    assert context.result["Prediction"] == expected_result


@then('it is based on {expected_features} features')
def step_impl(context, expected_features):
    assert context.result["Features"] == expected_features

    # mock_model.fit.assert_called
    # mock_model.predict.assert_called
