import os
from functools import wraps

import flask
from flask import request, json

USERNAME = 'fickle'

def Response(data, status = 200):
    body = json.dumps(data)
    return flask.Response(body, status = status, mimetype = 'application/json')

def SuccessResponse(dataset_id = None):
    return Response({ 'success': True, 'id': dataset_id })

def ErrorResponse(status = 400):
    return Response({ 'success': False }, status = status)

def check_auth(username, password):
    setting = os.environ.get('FICKLE_PASSWORD')
    if setting:
        return username == USERNAME and password == setting
    else:
        return True

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return ErrorResponse(403)
        return f(*args, **kwargs)
    return decorated

def API(name, backend):
    app = flask.Flask(name)
    app.config.from_object(name)

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
        return Response(data)

    @app.route('/predict', methods=['POST'])
    @requires_auth
    def api_predict():
        if not backend.trained():
            return ErrorResponse()
        data = backend.predict(request.json).tolist()
        return Response(data)

    return app
