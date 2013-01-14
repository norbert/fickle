import os
import pickle
import pylibmc

import sklearn.cross_validation

CACHE_KEY = 'fickle:predictor'
CACHE = pylibmc.Client(
        servers = [os.environ.get('MEMCACHE_SERVERS', '127.0.0.1')],
        username=os.environ.get('MEMCACHE_USERNAME'),
        password=os.environ.get('MEMCACHE_PASSWORD'),
        binary=True
)

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
        return bool(self.__model)

    def predict(self, value):
        if not self.trained() and not self.__load():
            return
        return self.__model.predict(value)

    def validate(self):
        if not self.loaded():
            return
        self.__random_id += 1
        if random_state is None:
            random_state = self.__random_id
        model = self.model()
        X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(
            self.__data, self.__target, test_size = None, random_state = self.__random_id
        )
        model.fit(X_train, y_train)
        return [model.score(X_test, y_test)]

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
