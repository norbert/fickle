from fickle import API

app = API(__name__)

if __name__ == '__main__':
    import os
    host = '0.0.0.0'
    port = int(os.environ.get('PORT', 5000))
    app.run(host=host, port=port)
