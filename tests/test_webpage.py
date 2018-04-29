import unittest
from webpage import app, db
from parameterized import parameterized
import os
from webpage.models import Currency
from markets.currency_analysis import CurrencyAnalyser
from unittest import mock


class FlaskTestCase(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        db_path = os.path.join(os.path.dirname(__file__), 'test.db')
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///{}'.format(db_path)
        app.config['SECRET_KEY'] = None
        app.config['WTF_CSRF_ENABLED'] = False

        self.client = app.test_client()

        db.session.commit()
        db.drop_all()
        db.create_all()

        c1 = Currency(name="USD", test_accuracy=80, train_accuracy=90, nr_features=123, nr_tweets=2345,
                      base_rate_accuracy=33)
        c2 = Currency(name="EUR", test_accuracy=60, train_accuracy=70, nr_features=321, nr_tweets=5432,
                      base_rate_accuracy=50)

        db.session.add(c1)
        db.session.add(c2)
        db.session.commit()

        app.currencies = Currency.get_all()

        dates = ['2018-03-06 11:22:33', '2018-03-07 22:33:44']
        tweets = ["First", "Second"]
        tweets_per_date = {'2018-03-06 11:22:33': ["First"], '2018-03-07 22:33:44': ["Second"]}
        for c in app.currencies:
            a = mock.create_autospec(CurrencyAnalyser)
            a.get_graph_data.return_value = (dates, tweets, tweets_per_date)
            a.get_most_coefficient_features.return_value = {"Down": [("Bad", 20), ("Taxes", 15)],
                                                            "NC": [("Apple", 20), ("Banana", 15)],
                                                            "Up": [("Good", 30), ("Nice", 15)]}
            a.get_rules_data.return_value = []
            a.analyse_tweet.return_value = {"Sentiment": "Positive", "Features": "F1, F2"}
            app.analysers[c.name] = a

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_index(self):
        response = self.client.get("/")
        self.assertEqual(200, response.status_code)
        self.assertIn(b'Tweets effect on USD', response.data)
        self.assertIn(b'<a class="nav-link " href="/currency/EUR">EUR</a>', response.data)
        self.assertIn(b'<a class="nav-link active" href="/">USD</a>', response.data)

    @parameterized.expand([('/currency/USD',), ('currency/USD',)])
    def test_currency_usd(self, path):
        response = self.client.get(path)
        self.assertEqual(301, response.status_code)
        self.assertEqual("http://localhost/", response.location)

    @parameterized.expand([('/currency/EUR',), ('currency/EUR',)])
    def test_currency_another(self, path):
        response = self.client.get(path)
        self.assertEqual(200, response.status_code)
        self.assertIn(b'Tweets effect on EUR', response.data)
        self.assertIn(b'<a class="nav-link active" href="/currency/EUR">EUR</a>', response.data)
        self.assertIn(b'<a class="nav-link " href="/">USD</a>', response.data)

    @parameterized.expand([('/currency/abc',), ('/usd',), ('/abc',), ('/currency',)])
    def test_404(self, wrong_path):
        response = self.client.get(wrong_path)
        self.assertEqual(404, response.status_code)

    def test_predictions(self):
        response = self.client.post('/', data={"tweet_content": "Some text"}, follow_redirects=True)
        self.assertEqual(200, response.status_code)
        self.assertIn(b'Sentiment : Positive', response.data)
        self.assertIn(b'Features : F1, F2', response.data)
        app.analysers["USD"].analyse_tweet.assert_called_once_with("Some text")

    @parameterized.expand([
        ('', b'This field is required.'),
        ('aa', b'Field must be between 3 and 300 characters long.'),
        ('aa' * 200, b'Field must be between 3 and 300 characters long.'),
    ])
    def test_predictions_wrong_length(self, tweet, expected_response):
        response = self.client.post('/', data={"tweet_content": tweet}, follow_redirects=True)
        self.assertEqual(200, response.status_code)
        self.assertIn(expected_response, response.data)
        app.analysers["USD"].analyse_tweet.assert_not_called()


# class PageLoadTest(unittest.TestCase):
#     def setUp(self):
#         self.driver = webdriver.Chrome()
#
#     def tearDown(self):
#         self.driver.quit()
#
#     def test_main_page(self):
#         self.driver.get('/')
#         element_to_check = self.driver.find_element_by_tag_name('h4')
#         self.assertEqual(element_to_check.text, 'Tweets affect on USD')
#


if __name__ == '__main__':
    unittest.main()
