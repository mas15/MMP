from webpage import db


class Currency(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(4), unique=True, nullable=False)
    full_name = db.Column(db.String(15), nullable=False)
    test_accuracy = db.Column(db.Integer, nullable=False)
    train_accuracy = db.Column(db.Integer, nullable=False)
   # zeroR = db.Column(db.Integer, nullable=False)
    # todo features, nr tweets itp?

    @staticmethod
    def get_currency(currency_name):
        return Currency.query.filter_by(name=currency_name).first()

    @staticmethod
    def get_all():
        return Currency.query.all()
