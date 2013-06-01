__all__ = ['Backend']

import os
import pickle
import redis

import sklearn.cross_validation

CACHE_KEY = 'fickle:model'
CACHE_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379')
CACHE = redis.from_url(CACHE_URL)


class Backend(object):

    def __init__(self):
        self.dataset_id = 0
        self._random_id = 0
        self._dataset = None
        self._model = None

    def load(self, dataset):
        self._model = None
        self.dataset_id += 1
        self._dataset = dataset
        self._data = dataset['data']
        self._target = dataset.get('target')
        return True

    def loaded(self):
        return self.dataset_id > 0

    def fit(self):
        if not self.loaded():
            return
        model = self.model()
        model.fit(self._data, self._target)
        self._model = model
        self._dump()
        return True

    def trained(self):
        return bool(self._model) or bool(self._load())

    def predict(self, value):
        if not self.trained():
            return
        return self._model.predict(value).tolist()

    def predict_probabilities(self, value):
        if not self.trained():
            return
        return self._model.predict_proba(value).tolist()

    def validate(self):
        if not self.loaded():
            return
        model = self.model()
        X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(
            self._data, self._target, random_state=self.random_id(True)
        )
        model.fit(X_train, y_train)
        return [model.score(X_test, y_test)]

    def random_id(self, increment=False):
        if bool(increment):
            self._random_id += 1
        return self._random_id

    def _load(self):
        string = CACHE.get(CACHE_KEY)
        if not string:
            return
        model = pickle.loads(string)
        self._model = model
        return True

    def _dump(self):
        if not self.trained():
            return
        string = pickle.dumps(self._model)
        CACHE.set(CACHE_KEY, string)
        return True
