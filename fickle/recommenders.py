__all__ = ['UserItemRecommender']

from scikits.crab.models import MatrixPreferenceDataModel
from scikits.crab.metrics import pearson_correlation
from scikits.crab.similarities import UserSimilarity
from scikits.crab.recommenders.knn.classes import UserBasedRecommender

from fickle.backend import Backend

HOW_MANY = 5


class UserItemRecommender(Backend):

    @staticmethod
    def model(*args, **kwargs):
        return UserBasedRecommender(*args, **kwargs)

    def load(self, dataset):
        self._model = None
        self.dataset_id += 1
        self._dataset = dataset
        self._data = MatrixPreferenceDataModel(dataset['data'])
        return True

    def fit(self):
        if not self.loaded():
            return
        similarity = UserSimilarity(self._data, pearson_correlation)
        recommender = self.model(self._data, similarity, with_preference=True)
        self._model = recommender
        return True

    def validate(self):
        pass

    def predict(self, value):
        if not self.trained():
            return
        if len(value) is not 1:
            return
        return self._model.recommend(value[0], how_many=HOW_MANY)

    def predict_probabilities(self, value):
        pass
