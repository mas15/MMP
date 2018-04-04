import unittest
from webpage import app


class FlaskTestCase(unittest.TestCase):

    def setUp(self):
        app.testing = True
        self.app = app.test_client()

    #def test_

if __name__ == '__main__':
    unittest.main()
