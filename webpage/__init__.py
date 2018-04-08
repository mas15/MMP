from flask import Flask
from markets.currency_analyser import CurrencyAnalyser

app = Flask(__name__, static_url_path='/static')
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['DEBUG'] = True
app.config['SECRET_KEY'] = b'\xdfP\xdb\xc9\xe4K\x0fc\x10\x06\xca\xaf\x1f\xb3\x00x\xc6\xd2\x96 lg\xf7\xad'

app.analysers = dict()
for c in ["USD", "EUR", "MEX"]:  # todo pobierac currencies
    a = CurrencyAnalyser(c)
    a.load()
    app.analysers[c] = a


def unzip(list_of_tuples):
    return zip(*list_of_tuples)


app.jinja_env.filters['unzip'] = unzip

import webpage.views

if __name__ == "__main__":
    app.run()
