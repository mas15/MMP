from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os

app = Flask(__name__, static_url_path='/static')
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['DEBUG'] = True
app.config['SECRET_KEY'] = b'\xdfP\xdb\xc9\xe4K\x0fc\x10\x06\xca\xaf\x1f\xb3\x00x\xc6\xd2\x96 lg\xf7\xad'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db_path = os.path.join(os.path.dirname(__file__), 'currencies.db')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///{}'.format(db_path)
db = SQLAlchemy(app)

app.analysers = dict()
app.currencies = []


def unzip(list_of_tuples):
    return zip(*list_of_tuples)


app.jinja_env.filters['unzip'] = unzip

import webpage.views
import webpage.models
