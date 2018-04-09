# from https://flask-script.readthedocs.io/en/latest/
from webpage import app, db
from flask_script import Manager, prompt_bool
from webpage.models import Currency
from markets.currency_analyser import CurrencyAnalyser

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
    for c in ["EUR"]: #"USD", , "MEX"]:
        analyser = CurrencyAnalyser(c)
        test_accuracy, train_accuracy = analyser.analyse()
        currency = Currency(name=c, full_name=c, test_accuracy=test_accuracy, train_accuracy=train_accuracy)
        db.session.add(currency)
        db.session.commit()


if __name__ == '__main__':
    manager.run()
