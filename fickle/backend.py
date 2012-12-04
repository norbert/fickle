import sklearn.cross_validation

class Backend(object):
    def __init__(self):
        self.dataset_id = 0
        self.random_id = 0
        self.dataset = None
        self.model = None

    def load(self, dataset):
        self.model = None
        self.dataset_id += 1
        self.dataset = dataset
        self._data = dataset['data']
        self._target = dataset['target']
        return True

    def loaded(self):
        return (self.dataset is not None)

    def fit(self):
        if not self.loaded():
            return False
        model = self.classifier()
        model.fit(self._data, self._target)
        self.model = model
        return True

    def trained(self):
        return (self.model is not None)

    def validate(self, test_size = 0.2):
        if not self.loaded():
            return
        self.random_id += 1
        model = self.classifier()
        X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(
            self._data, self._target, test_size = test_size, random_state = self.random_id
        )
        model.fit(X_train, y_train)
        return [model.score(X_test, y_test)]

    def predict(self, value):
        return self.model.predict(value)
