from webpage import db


class Currency(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(4), unique=True, nullable=False)
    test_accuracy = db.Column(db.Integer, nullable=False)
    train_accuracy = db.Column(db.Integer, nullable=False)
    zero_r = db.Column(db.Integer, nullable=False)
    nr_tweets = db.Column(db.Integer, nullable=False)
    nr_features = db.Column(db.Integer, nullable=False)

    @staticmethod
    def get_currency(currency_name):
        return Currency.query.filter_by(name=currency_name).first()

    @staticmethod
    def get_all():
        return Currency.query.all()
