from flask import Flask, abort
from flask import request
from flask_cors import *
from gevent.pywsgi import WSGIServer
from detect import detect

app = Flask(__name__)


@app.route('/request', methods=['POST'])
@cross_origin()
def classify():
    if not request.json or 'path' not in request.json:
        abort(400)
    path = request.json['path']
    predictions = detect(path)

    return {'predictions': predictions}


if __name__ == '__main__':
    app.run()
    # http_server = WSGIServer(('', 5000), app)
    # http_server.serve_forever()
