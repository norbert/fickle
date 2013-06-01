__all__ = ['API']

import os
from functools import wraps

import flask
from flask import request, json

import predictors
import recommenders

models = (predictors, recommenders)

USERNAME = 'fickle'


def Response(data=None, status=200):
    if data:
        body = json.dumps(data)
    else:
        body = None
    return flask.Response(body, status=status, mimetype='application/json')


def SuccessResponse(dataset_id=None):
    return Response({'success': True, 'id': dataset_id})


def ErrorResponse(status=400):
    return Response(status=status)


def API(name, backend=None):
    app = flask.Flask(name)
    app.config['DEBUG'] = bool(os.environ.get('FICKLE_DEBUG'))

    if backend is None:
        __model = os.environ.get('FICKLE_MODEL', 'GenericSVMClassifier')
        model = next((getattr(m, __model) for m in models
                      if hasattr(m, __model)))
        backend = model()

    __password = os.environ.get('FICKLE_PASSWORD')

    def check_auth(username, password):
        if __password:
            return username == USERNAME and password == __password
        else:
            return True

    def requires_auth(f):
        if not __password:
            return f

        @wraps(f)
        def decorated(*args, **kwargs):
            auth = request.authorization
            if not auth or not check_auth(auth.username, auth.password):
                return ErrorResponse(403)
            return f(*args, **kwargs)
        return decorated

    @app.route('/')
    @requires_auth
    def api_root():
        return SuccessResponse(backend.dataset_id)

    @app.route('/load', methods=['POST'])
    @requires_auth
    def api_load():
        backend.load(request.json)
        return SuccessResponse(backend.dataset_id)

    @app.route('/fit', methods=['POST'])
    @requires_auth
    def api_fit():
        if not backend.loaded():
            return ErrorResponse()
        backend.fit()
        return SuccessResponse(backend.dataset_id)

    @app.route('/validate', methods=['POST'])
    @requires_auth
    def api_validate():
        if not backend.loaded():
            return ErrorResponse()
        data = backend.validate()
        if data is None:
            return ErrorResponse()
        return Response(data)

    @app.route('/predict', methods=['POST'])
    @requires_auth
    def api_predict():
        if not backend.trained():
            return ErrorResponse()
        data = backend.predict(request.json)
        if data is None:
            return ErrorResponse()
        return Response(data)

    @app.route('/predict/probabilities', methods=['POST'])
    @requires_auth
    def api_predict_probabilities():
        if not backend.trained():
            return ErrorResponse()
        if data is None:
            return ErrorResponse()
        data = backend.predict_probabilities(request.json)
        return Response(data)

    return app
