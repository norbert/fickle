from sklearn import datasets

from fickle.testing import TestCase
from fickle.predictors import GenericSVMClassifier as Backend


class BackendTest(TestCase):

    def test_load(self):
        backend = Backend()
        dataset = datasets.load_iris()
        self.assertTrue(backend.load(dataset))
        self.assertTrue(backend.load(dataset))

    def test_loaded(self):
        backend = Backend()
        dataset = datasets.load_iris()
        self.assertFalse(backend.loaded())
        backend.load(dataset)
        self.assertTrue(backend.loaded())
        backend.load(dataset)
        self.assertTrue(backend.loaded())

    def test_fit_when_not_loaded(self):
        backend = Backend()
        with self.assertRaises(RuntimeError):
            backend.fit()

    def test_fit_when_loaded(self):
        backend = Backend()
        dataset = datasets.load_iris()
        backend.load(dataset)
        self.assertTrue(backend.fit())

    def test_trained_without_load(self):
        backend = Backend()
        dataset = datasets.load_iris()
        self.assertFalse(backend.trained())
        backend.load(dataset)
        self.assertFalse(backend.trained())
        backend.fit()
        self.assertTrue(backend.trained())

    def test_trained_with_load(self):
        old_backend = Backend()
        dataset = datasets.load_iris()
        old_backend.load(dataset)
        old_backend.fit()
        new_backend = Backend()
        self.assertTrue(new_backend.trained(read=True))

    def test_predict_when_trained(self):
        backend = Backend()
        dataset = datasets.load_iris()
        backend.load(dataset)
        backend.fit()
        sample = dataset['data'][:10]
        predictions = backend.predict(sample)
        self.assertEqual(len(predictions), 10)

    def test_predict_probabilities_when_trained(self):
        backend = Backend()
        dataset = datasets.load_iris()
        backend.load(dataset)
        backend.fit()
        sample = dataset['data'][:10]
        with self.assertRaises(AttributeError):
            backend.predict_probabilities(sample)

    def test_validate_when_loaded(self):
        backend = Backend()
        dataset = datasets.load_iris()
        backend.load(dataset)
        score1 = backend.validate()
        self.assertEqual(len(score1), 1)
        self.assertEqual(backend.random_id(), 1)
        score2 = backend.validate()
        self.assertEqual(backend.random_id(), 2)
        self.assertNotEqual(score1[0], score2[0])
        self.assertGreater(score1[0], 0.8)
        self.assertGreater(score2[0], 0.8)
