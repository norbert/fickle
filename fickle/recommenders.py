__all__ = ['UserItemRecommender']

try:
    from recsys.datamodel.data import Data
    from recsys.algorithm.factorize import SVD
except ImportError:
    pass

from .backend import Backend

K = 100
MIN_VALUES = 2
N = 10


class Recommender(Backend):

    def load(self, dataset, write=None):
        self._model = None
        data = self._convert_hash(dataset['data'])
        self._data = data
        if write:
            pass
        return True

    def predict(self, keys):
        return self.recommend(keys)

    def recommend(self, keys, n=None, unknown=True):
        if n is None:
            n = N
        self._ensure_trained(load=True)
        if type(keys) not in (list, tuple):
            raise ValueError
        elif len(keys) < 1:
            raise ValueError
        model = self._model

        def call(key):
            try:
                return model.recommend(key, n=n, only_unknowns=unknown)
            except KeyError:
                pass
        return map(call, keys)

    def validate(self):
        pass

    def predict_probabilities(self, value):
        pass

    def _convert_hash(self, dataset):
        data = Data()
        for key in dataset:
            record = dataset[key]
            batch = [(record[k], key, k) for k in record]
            data.set(batch, extend=True)
        return data


class UserItemRecommender(Recommender):

    @staticmethod
    def model(*args, **kwargs):
        return SVD()

    def fit(self, k=K):
        self._ensure_loaded()
        model = self.model()
        model.set_data(self._data)
        matrix = model.create_matrix()
        self.matrix = matrix
        model.compute(k=k, min_values=MIN_VALUES,
                      mean_center=True, post_normalize=True)
        model.set_data(Data())
        self._model = model
        self._dump_model()
        return True
