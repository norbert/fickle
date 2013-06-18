# Fickle

Experimental machine learning REST API.

## Summary

This package contains modules for building a machine learning or recommender system using [scikit-learn](http://scikit-learn.org/) and [python-recsys](https://github.com/ocelma/python-recsys) for use in a [Flask](http://flask.pocoo.org/) [WSGI](http://www.python.org/dev/peps/pep-3333/) application.

## Configuration

All configuration is done using environment variables. The following variables are recognized:

* `PORT`: Web server port. Defaults to 5000.
* `FICKLE_MODEL`: Predictor or recommender model to use. Defaults to `GenericSVMClassifier`.
* `FICKLE_DEBUG`: Web server debug mode. Defaults to false.
* `FICKLE_PASSWORD`: Basic authentication password. Authentication is disabled when empty.
* `REDIS_URL`: Storage layer connection URL. Defaults to local instance.

## Requirements

See `requirements.txt` for details.

## Deployment

### Heroku

```
$ git clone https://github.com/norbert/fickle && cd fickle
$ heroku apps:create -b git://github.com/norbert/heroku-buildpack-python.git#numpy
...
$ git push heroku master
...
$ heroku run ipython
```
