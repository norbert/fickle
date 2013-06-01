from flask.ext.testing import TestCase as FlaskTestCase
from fickle.testing import TestCase as FickleTestCase
from mock import Mock
import json
from fickle.api import API
from fickle.classifier import GenericSVMClassifier as Backend
from sklearn import datasets


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

    def assert_success(self, response, _id):
        self.assert200(response)
        value = {'id': _id, 'success': True}
        self.assertEqual(response.json, value)

    def test_root(self):
        response = self.get('/')
        self.assert_success(response, 0)

    def test_load(self):
        dataset = datasets.load_iris()
        response = self.load(dataset)
        self.assert_success(response, 1)
        self.assertTrue(self.backend.loaded())

    def test_fit_when_not_loaded(self):
        response = self.post('/fit')
        self.assert400(response)

    def test_fit_when_loaded(self):
        dataset = datasets.load_iris()
        self.load(dataset)
        response = self.post('/fit')
        self.assert_success(response, 1)
        self.assertTrue(self.backend.trained())

    def test_validate_when_loaded(self):
        dataset = datasets.load_iris()
        self.load(dataset)
        response = self.post('/validate')
        self.assert200(response)
        self.assertEqual(len(response.json), 1)
        self.assertGreater(response.json[0], 0.8)
        self.assertEqual(self.backend.random_id(), 1)

    def test_predict_when_trained(self):
        dataset = datasets.load_iris()
        self.load(dataset)
        self.post('/fit')
        sample = dataset['data'][:10]
        response = self.post('/predict', data=sample.tolist())
        self.assert200(response)
        self.assertEqual(len(response.json), 10)
        self.assertEqual(response.json, self.backend.predict(sample))
