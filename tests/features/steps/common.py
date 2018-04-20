from behave import given, when, then, step


@given('we have texts: {texts}')
def step_impl(context, texts):
    context.texts = texts.split(", ")
