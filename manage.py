# from https://flask-script.readthedocs.io/en/latest/
from webpage import app, db
from flask_script import Manager, prompt_bool
from webpage.models import Currency
from markets.currency_analysis import CurrencyAnalyser

manager = Manager(app)


@manager.command
def run():
    app.currencies = Currency.get_all()
    for c in app.currencies:
        a = CurrencyAnalyser(c.name)
        a.load()
        app.analysers[c.name] = a
    app.run()


@manager.command
def initdb():
    db.create_all()


@manager.command
def dropdb():
    if prompt_bool("Are you sure you want to lose all your data"):
        db.drop_all()


@manager.command
def demo():
    for c in ["USD", "EUR", "MEX"]:
        analyser = CurrencyAnalyser(c)
        analyse_result = analyser.analyse()
        currency = Currency(name=c,
                            test_accuracy=analyse_result.test_accuracy,
                            train_accuracy=analyse_result.train_accuracy,
                            nr_features=analyse_result.nr_features,
                            nr_tweets=analyse_result.nr_tweets,
                            base_rate_accuracy=analyse_result.base_rate_accuracy)
        db.session.add(currency)
        db.session.commit()


if __name__ == '__main__':
    manager.run()
