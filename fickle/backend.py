__all__ = ['Backend']

from sklearn.cross_validation import train_test_split

from .storage import *


class Backend(object):

    def __init__(self):
        self._data = None
        self._model = None

    def load(self, dataset, write=False):
        self._model = None
        self._data = dataset['data']
        self._target = dataset.get('target')
        if write:
            self._write_dataset(dataset)
        return True

    def isloaded(self, read=False):
        if getattr(self, '_data', None) is not None:
            return True
        elif read:
            return bool(self._read_dataset())
        else:
            return False

    def fit(self, write=True):
        self._ensure_loaded(read=True)
        model = self.model()
        model.fit(self._data, self._target)
        self._model = model
        if write:
            self._write_model()
        return True

    def istrained(self, read=False):
        if getattr(self, '_model', None) is not None:
            return True
        elif read:
            return bool(self._read_model())
        else:
            return False

    def predict(self, value):
        self._ensure_trained(read=True)
        return self._model.predict(value).tolist()

    def predict_probabilities(self, value):
        self._ensure_trained(read=True)
        return self._model.predict_proba(value).tolist()

    def _read_dataset(self):
        dataset = read_key(DATASET_KEY)
        if dataset is not None:
            self.load(dataset, write=False)
        else:
            return False

    def _write_dataset(self, dataset):
        self._ensure_trained()
        write_key(DATASET_KEY, dataset)

    def _read_model(self):
        self._model = read_key(MODEL_KEY)
        return bool(self._model)

    def _write_model(self):
        self._ensure_trained()
        write_key(MODEL_KEY, self._model)

    def _ensure_loaded(self, read=False):
        if not self.isloaded(read):
            raise RuntimeError

    def _ensure_trained(self, read=False):
        if not self.istrained(read):
            raise RuntimeError
