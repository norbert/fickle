import sklearn.cross_validation

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
        return (self.__dataset is not None)

    def fit(self):
        if not self.loaded():
            return False
        model = self.model()
        model.fit(self.__data, self.__target)
        self.__model = model
        return True

    def trained(self):
        return (self.__model is not None)

    def predict(self, value):
        if not self.trained():
            return
        return self.__model.predict(value)

    def validate(self, test_size = 0.2, random_state = None):
        if not self.loaded():
            return
        self.__random_id += 1
        if random_state is None:
            random_state = self.__random_id
        model = self.model()
        X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(
            self.__data, self.__target, test_size = test_size, random_state = self.__random_id
        )
        model.fit(X_train, y_train)
        return [model.score(X_test, y_test)]
