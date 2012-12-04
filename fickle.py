from flask import Flask, request, Response, json, g
from sklearn import cross_validation

class Backend(object):
    def __init__(self):
        self.dataset_id = 0
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

    def validate(self, test_size = 0.2, random_state = 0):
        if not self.loaded():
            return
        model = self.classifier()
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            self._data, self._target, test_size = test_size, random_state = random_state
        )
        model.fit(X_train, y_train)
        return [model.score(X_test, y_test)]

    def predict(self, value):
        return self.model.predict(value)

def json_response(data, status = 200):
    body = json.dumps(data)
    return Response(body, status = status, mimetype = 'application/json')

def success_response(dataset_id = None):
    return json_response({ 'success': True, 'id': dataset_id })

def error_response(status = 400):
    return json_response({ 'success': False }, status = status)

def API(name, backend):
    app = Flask(name)
    app.config.from_object(name)

    @app.route('/')
    def api_root():
        return success_response(backend.dataset_id)

    @app.route('/load', methods=['POST'])
    def api_load():
        backend.load(request.json)
        return success_response(backend.dataset_id)

    @app.route('/fit', methods=['POST'])
    def api_fit():
        if not backend.loaded():
            return error_response()
        backend.fit()
        return success_response(backend.dataset_id)

    @app.route('/validate', methods=['POST'])
    def api_validate():
        if not backend.loaded():
            return error_response()
        data = backend.validate()
        return json_response(data)

    @app.route('/predict', methods=['PUT'])
    def api_predict():
        if not backend.trained():
            return error_response()
        data = backend.predict(request.json).tolist()
        return json_response(data)

    return app
