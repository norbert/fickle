__all__ = ['Backend']

import os
import pickle
import redis

import sklearn.cross_validation

CACHE_KEY = 'fickle:predictor'
CACHE_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379')
CACHE = redis.from_url(CACHE_URL)

class Backend(object):
    def __init__(self):
        self.dataset_id = 0
        self.__random_id = 0
        self.__dataset = None
        self.__model = None

    def load(self, dataset):
        self.__model = None
        self.dataset_id += 1
        self.__dataset = dataset
        self.__data = dataset['data']
        self.__target = dataset['target']
        return True

    def loaded(self):
        return bool(self.__dataset)

    def fit(self):
        if not self.loaded():
            return
        model = self.model()
        model.fit(self.__data, self.__target)
        self.__model = model
        self.__dump()
        return True

    def trained(self):
        return bool(self.__model) or bool(self.__load())

    def predict(self, value):
        if not self.trained():
            return
        return self.__model.predict(value)

    def predict_probabilities(self, value):
        if not self.trained():
            return
        return self.__model.predict_proba(value)

    def validate(self):
        if not self.loaded():
            return
        model = self.model()
        X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(
            self.__data, self.__target, random_state = self.random_id(True)
        )
        model.fit(X_train, y_train)
        return [model.score(X_test, y_test)]

    def random_id(self, increment = False):
        if bool(increment):
            self.__random_id += 1
        return self.__random_id

    def __load(self):
        string = CACHE.get(CACHE_KEY)
        if not string:
            return
        model = pickle.loads(string)
        self.__model = model
        return True

    def __dump(self):
        if not self.trained():
            return
        string = pickle.dumps(self.__model)
        CACHE.set(CACHE_KEY, string)
        return True
