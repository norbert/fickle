import json

from mock import Mock
from flask.ext.testing import TestCase as FlaskTestCase
from sklearn import datasets

from fickle.testing import TestCase as FickleTestCase
from fickle.api import API
from fickle.predictors import GenericSVMClassifier as Backend


class APITest(FlaskTestCase, FickleTestCase):

    def create_app(self):
        self.backend = Backend()
        app = API(__name__, backend=self.backend)
        app.config['TESTING'] = True
        return app

    def get(self, *args, **kwargs):
        return self.client.get(*args, **kwargs)

    def post(self, *args, **kwargs):
        kwargs['content_type'] = 'application/json'
        if kwargs.get('data'):
            kwargs['data'] = json.dumps(kwargs['data'])
        return self.client.post(*args, **kwargs)

    def load(self, dataset):
        return self.post('/load', data={
            'data': dataset.data.tolist(),
            'target': dataset.target.tolist()
        })

    def assert_success(self, response, status=200):
        self.assertEqual(response.status_code, status)

    def assert_error(self, response, status=400):
        self.assertEqual(response.status_code, status)

    def test_root(self):
        response = self.get('/')
        self.assert_success(response)

    def test_load(self):
        dataset = datasets.load_iris()
        response = self.load(dataset)
        self.assert_success(response, status=201)
        self.assertTrue(self.backend.isloaded())

    def test_fit_when_not_loaded(self):
        response = self.post('/fit')
        self.assert_error(response, status=501)

    def test_fit_when_loaded(self):
        dataset = datasets.load_iris()
        self.load(dataset)
        response = self.post('/fit')
        self.assert_success(response)
        self.assertTrue(self.backend.istrained())

    def test_predict_when_trained(self):
        dataset = datasets.load_iris()
        self.load(dataset)
        self.post('/fit')
        sample = dataset['data'][:10]
        response = self.post('/predict', data=sample.tolist())
        self.assert200(response)
        self.assertEqual(len(response.json), 10)
        self.assertEqual(response.json, self.backend.predict(sample))
