from behave import given


@given('we have texts: {texts}')
def step_impl(context, texts):
    context.texts = texts.split(", ")
