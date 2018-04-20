import os
import sys
sys.path.insert(0, os.path.pardir)

from behave import given, when, then, step
from markets.phrases_extractor import PhrasesExtractor


@given('we have vocabulary of: {vocabulary}')
def step_impl(context, vocabulary):
    context.e = PhrasesExtractor()
    context.e.set_features(vocabulary.split(", "))


@when('we extract phrases from the sentence: {sentence}')
def step_impl(context, sentence):
    context.e.result = context.e.extract_features(sentence)


@then('we get {phrases} extracted')
def step_impl(context, phrases):
    print(context.e.result )
    assert context.e.result == phrases.split(", ")
