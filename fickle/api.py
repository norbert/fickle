import flask
from flask import request, json

def Response(data, status = 200):
    body = json.dumps(data)
    return flask.Response(body, status = status, mimetype = 'application/json')

def SuccessResponse(dataset_id = None):
    return Response({ 'success': True, 'id': dataset_id })

def ErrorResponse(status = 400):
    return Response({ 'success': False }, status = status)

def API(name, backend):
    app = flask.Flask(name)
    app.config.from_object(name)

    @app.route('/')
    def api_root():
        return SuccessResponse(backend.dataset_id)

    @app.route('/load', methods=['POST'])
    def api_load():
        backend.load(request.json)
        return SuccessResponse(backend.dataset_id)

    @app.route('/fit', methods=['POST'])
    def api_fit():
        if not backend.loaded():
            return ErrorResponse()
        backend.fit()
        return SuccessResponse(backend.dataset_id)

    @app.route('/validate', methods=['PUT'])
    def api_validate():
        if not backend.loaded():
            return ErrorResponse()
        data = backend.validate()
        return Response(data)

    @app.route('/predict', methods=['POST'])
    def api_predict():
        if not backend.trained():
            return ErrorResponse()
        data = backend.predict(request.json).tolist()
        return Response(data)

    return app
