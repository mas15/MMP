import os
import sys
sys.path.insert(0, os.path.pardir)

from behave import given, when, then, step
from markets.sentiment import SentimentAnalyser


@given('we created a sentiment analyser')
def step_impl(context):
    context.s = SentimentAnalyser()


@given('we know their sentiment: {sentiments}')
def step_impl(context, sentiments):
    context.texts = list(zip(context.texts, sentiments.split(", ")))


@when('we train a sentiment analyser')
def step_impl(context):
    context.s.train(context.texts)


@then('we can predict that {text} is {expected_result}')
def step_impl(context, text, expected_result):
    result = context.s.predict(text)
    expected_result = "pos" if expected_result == "positive" else "neg"
    assert expected_result == result
