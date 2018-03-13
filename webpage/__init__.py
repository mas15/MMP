from flask import Flask
from markets.sentiment import SentimentAnalyser

app = Flask(__name__, static_url_path='/static')
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['DEBUG'] = True
app.config['SECRET_KEY'] = b'\xdfP\xdb\xc9\xe4K\x0fc\x10\x06\xca\xaf\x1f\xb3\x00x\xc6\xd2\x96 lg\xf7\xad'

app.sent = SentimentAnalyser()

import webpage.views

if __name__ == "__main__":
    app.run()
