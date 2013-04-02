from fickle import API

from fickle.classifier import *
backend = GenericSVMClassifier()

app = API(__name__, backend)

if __name__ == '__main__':
    import os
    host = '0.0.0.0'
    port = int(os.environ.get('PORT', 5000))
    debug = bool(os.environ.get('FICKLE_DEBUG'))
    app.run(host = host, port = port, debug = debug)
