__all__ = ['Backend']

import pickle
import zlib

from sklearn.cross_validation import train_test_split

from cache import *


class Backend(object):

    def __init__(self):
        self._random_id = 0
        self._data = None
        self._model = None

    def load(self, dataset):
        self._model = None
        self._dataset = dataset
        self._data = dataset['data']
        self._target = dataset.get('target')
        return True

    def loaded(self):
        return self._data is not None

    def fit(self):
        self._ensure_loaded()
        model = self.model()
        model.fit(self._data, self._target)
        self._model = model
        self._dump_model()
        return True

    def trained(self, load=False):
        if getattr(self, '_model', None) is not None:
            return True
        elif load:
            return bool(self._load_model())
        else:
            return False

    def predict(self, value):
        self._ensure_trained(load=True)
        return self._model.predict(value).tolist()

    def predict_probabilities(self, value):
        self._ensure_trained(load=True)
        return self._model.predict_proba(value).tolist()

    def validate(self):
        self._ensure_loaded()
        model = self.model()
        X_train, X_test, y_train, y_test = train_test_split(
            self._data, self._target, random_state=self.random_id(True)
        )
        model.fit(X_train, y_train)
        return [model.score(X_test, y_test)]

    def random_id(self, increment=False):
        if bool(increment):
            self._random_id += 1
        return self._random_id

    def _load_model(self):
        string = CACHE.get(CACHE_KEY)
        if not string:
            return False
        model = pickle.loads(zlib.decompress(string))
        self._model = model
        return True

    def _dump_model(self):
        self._ensure_trained()
        string = zlib.compress(pickle.dumps(self._model))
        CACHE.set(CACHE_KEY, string)
        return True

    def _ensure_loaded(self):
        if not self.loaded():
            raise RuntimeError

    def _ensure_trained(self, load=False):
        if not self.trained(load=load):
            raise RuntimeError
