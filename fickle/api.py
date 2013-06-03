__all__ = ['API']

from functools import wraps
import os

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


def SuccessResponse(status=200, dataset_id=None):
    return Response(None, status=status)


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
                return ErrorResponse(status=403)
            return f(*args, **kwargs)
        return decorated

    @app.route('/')
    @requires_auth
    def api_root():
        return SuccessResponse()

    @app.route('/load', methods=['POST'])
    @requires_auth
    def api_load():
        backend.load(request.json)
        return SuccessResponse(status=201)

    @app.route('/fit', methods=['POST'])
    @requires_auth
    def api_fit():
        try:
            backend.fit()
        except RuntimeError:
            return ErrorResponse(status=501)
        return SuccessResponse()

    @app.route('/validate', methods=['POST'])
    @requires_auth
    def api_validate():
        if not backend.loaded():
            return ErrorResponse(status=501)
        data = backend.validate()
        return Response(data)

    @app.route('/predict', methods=['POST'])
    @requires_auth
    def api_predict():
        return api_predict_method('predict')

    @app.route('/predict/probabilities', methods=['POST'])
    @requires_auth
    def api_predict_probabilities():
        return api_predict_method('predict_probabilities')

    def api_predict_method(method):
        try:
            data = getattr(backend, method)(request.json)
        except RuntimeError:
            return ErrorResponse(status=501)
        return Response(data)

    @app.route('/recommend', methods=['POST'])
    @requires_auth
    def api_recommend():
        if not hasattr(backend, 'recommend'):
            return ErrorResponse(status=406)
        keys = request.json
        args = dict()
        n = request.args.get('n', None)
        if n is not None:
            args['n'] = int(n)
        try:
            data = backend.recommend(keys, **args)
        except RuntimeError:
            return ErrorResponse(status=500)
        except ValueError:
            return ErrorResponse(status=400)
        return Response(data)

    return app
