import io
import os

import tensorflowjs as tfjs
import werkzeug.formparser
from flask import Flask, Response, request
from flask_cors import CORS, cross_origin

tmpdir = os.getenv('TMPDIR')
full = os.getenv('FULL')


class ModelReceiver(object):

    def __init__(self):
        self._model = None
        self._model_json_bytes = None
        self._model_json_writer = None
        self._weight_bytes = None
        self._weight_writer = None

    @property
    def model(self):
        self._model_json_writer.flush()
        self._weight_writer.flush()
        self._model_json_writer.seek(0)
        self._weight_writer.seek(0)
        json_content = self._model_json_bytes.read()
        weights_content = self._weight_bytes.read()
        return tfjs.converters.deserialize_keras_model(json_content, weight_data=[weights_content])

    def stream_factory(self, total_content_length, content_type, filename, content_length=None):
        if filename == 'model.json':
            self._model_json_bytes = io.BytesIO()
            self._model_json_writer = io.BufferedWriter(self._model_json_bytes)
            return self._model_json_writer
        elif filename == 'model.weights.bin':
            self._weight_bytes = io.BytesIO()
            self._weight_writer = io.BufferedWriter(self._weight_bytes)
            return self._weight_writer


def main():
    app = Flask('model-server')
    CORS(app)
    app.config['CORS_HEADER'] = 'Content-Type'
    model_receiver = ModelReceiver()

    @app.route('/upload', methods=['POST'])
    @cross_origin()
    def upload():
        werkzeug.formparser.parse_form_data(request.environ, stream_factory=model_receiver.stream_factory)
        model = model_receiver.model
        model.summary()
        # You can perform `model.predict()`, `model.fit()`,
        # `model.evaluate()` etc. here.
        return Response(status=200)
    app.run('0.0.0.0', 5000)


if __name__ == '__main__':
    main()
