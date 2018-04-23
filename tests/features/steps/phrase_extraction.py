import os
import sys
sys.path.insert(0, os.path.pardir)

from behave import given, when, then
from markets.phrases_extraction import PhrasesExtractor


@given('we created a phrases extractor')
def step_impl(context):
    context.e = PhrasesExtractor()


@given('we have vocabulary of: {vocabulary}')
def step_impl(context, vocabulary):
    context.e.set_features(vocabulary.split(", "))


@when('we extract phrases from the sentence: {sentence}')
def step_impl(context, sentence):
    context.e.result = context.e.extract_features(sentence)


@when('we build a vocabulary')
def step_impl(context):
    context.e.build(context.texts)


@then('we get {phrases} extracted')
def step_impl(context, phrases):
    exp_result = set(phrases.split(", "))
    result = set([k for k, v in context.e.result.items() if v])
    assert exp_result == result


@then('nothing is extracted')
def step_impl(context):
    result = [k for k, v in context.e.result.items() if v]
    assert len(result) == 0


@then('the vocabulary consists of {phrases}')
def step_impl(context, phrases):
    exp_result = set(phrases.split(", "))
    result = set(context.e.features)
    assert exp_result == result
