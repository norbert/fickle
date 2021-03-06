from sklearn import datasets

from fickle.testing import TestCase
from fickle.predictors import GenericSVMClassifier as Backend


class BackendTest(TestCase):

    def test_load(self):
        backend = Backend()
        dataset = datasets.load_iris()
        self.assertTrue(backend.load(dataset))

    def test_isloaded(self):
        backend = Backend()
        dataset = datasets.load_iris()
        self.assertFalse(backend.isloaded())
        backend.load(dataset)
        self.assertTrue(backend.isloaded())
        backend.load(dataset)
        self.assertTrue(backend.isloaded())

    def test_fit_when_not_loaded(self):
        backend = Backend()
        with self.assertRaises(RuntimeError):
            backend.fit()

    def test_fit_when_loaded(self):
        backend = Backend()
        dataset = datasets.load_iris()
        backend.load(dataset)
        self.assertTrue(backend.fit())

    def test_istrained_without_load(self):
        backend = Backend()
        dataset = datasets.load_iris()
        self.assertFalse(backend.istrained())
        backend.load(dataset)
        self.assertFalse(backend.istrained())
        backend.fit()
        self.assertTrue(backend.istrained())

    def test_istrained_with_load(self):
        old_backend = Backend()
        dataset = datasets.load_iris()
        old_backend.load(dataset)
        old_backend.fit()
        new_backend = Backend()
        self.assertTrue(new_backend.istrained(read=True))

    def test_predict_when_trained(self):
        backend = Backend()
        dataset = datasets.load_iris()
        backend.load(dataset)
        backend.fit()
        sample = dataset['data'][:10]
        predictions = backend.predict(sample)
        self.assertEqual(len(predictions), 10)
