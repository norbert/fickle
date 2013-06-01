__all__ = ['GenericSVMClassifier', 'NearestNeighbors']

import numpy as np
import scipy.sparse as sp
from sklearn import svm, neighbors, preprocessing

from fickle.backend import Backend


class GenericSVMClassifier(Backend):

    @staticmethod
    def model():
        return svm.LinearSVC()


class NearestNeighbors(Backend):

    @staticmethod
    def model():
        return neighbors.NearestNeighbors()

    def load(self, dataset):
        self._model = None
        self.dataset_id += 1
        self._dataset = dataset
        X = dataset['data']
        Uk, Ik, Rk = range(3)
        UE = self._encode(X, Uk)
        IE = self._encode(X, Ik)
        Un = len(UE.classes_)
        In = len(IE.classes_)
        dtype = np.uint8
        UX = np.zeros((Un, In), dtype=dtype)
        IX = np.zeros((In, Un), dtype=dtype)
        for d in X:
            v = d[Rk]
            if v > 0:
                i = UE.transform([d[Uk]])[0]
                j = IE.transform([d[Ik]])[0]
                UX[i, j] = v
                IX[j, i] = v
        self._data = [sp.csr_matrix(UX), sp.csr_matrix(IX)]
        self._encoders = [UE, IE]
        return True

    def fit(self):
        if not self.loaded():
            return
        UM = self.model()
        IM = self.model()
        UM.fit(self._data[0])
        IM.fit(self._data[1])
        models = [UM, IM]
        self._model = [self._data, self._encoders, models]
        self._dump()
        return True

    def validate(self):
        pass

    def predict(self, value):
        if not self.trained():
            return
        if len(value) is not 2:
            return

        def kneighbors(index, label):
            if label is None:
                return
            X, E, M = [self._model[x][index] for x in range(3)]
            Xi = E.transform([label])[0]
            ds, ns = M.kneighbors(X[Xi])
            return [list(E.inverse_transform(ns[0])), list(ds[0])]
        return [kneighbors(i, l) for i, l in enumerate(value)]

    def predict_probabilities(self, value):
        pass

    def _encode(self, data, key):
        labels = [d[key] for d in data]
        encoder = preprocessing.LabelEncoder()
        encoder.fit(labels)
        return encoder
