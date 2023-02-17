import io
from os import environ

import tensorflowjs as tfjs
from flask import Flask, Response, request
from flask_cors import CORS, cross_origin
from werkzeug import formparser


class ModelReceiver():

    def stream_factory(self: 'ModelReceiver', total_content_length, content_type, filename, content_length=None): # type: ignore[no-untyped-def] # noqa: ANN001, ANN201, ARG002
        if filename == 'model.json':
            self.model_json_bytes = io.BytesIO()
            self.model_json_writer = io.BufferedWriter(self.model_json_bytes) # type: ignore[arg-type]
            return self.model_json_writer
        if filename == 'model.weights.bin':
            self.weight_bytes = io.BytesIO()
            self.weight_writer = io.BufferedWriter(self.weight_bytes) # type: ignore[arg-type]
            return self.weight_writer
        return None


def main() -> None:
    app = Flask('model-server')
    CORS(app)
    app.config['CORS_HEADER'] = 'Content-Type'
    model_receiver = ModelReceiver()

    @app.route('/upload', methods=['POST'])
    @cross_origin() # type: ignore[misc]
    def upload() -> Response:
        formparser.parse_form_data(request.environ, stream_factory=model_receiver.stream_factory)
        model_receiver.model_json_writer.flush()
        model_receiver.weight_writer.flush()
        model_receiver.model_json_writer.seek(0)
        model_receiver.weight_writer.seek(0)
        json_content = model_receiver.model_json_bytes.read()
        weights_content = model_receiver.weight_bytes.read()
        model = tfjs.converters.deserialize_keras_model(json_content, weight_data=[weights_content])
        model.summary()
        # You can perform `model.predict()`, `model.fit()`,
        # `model.evaluate()` etc. here.
        return Response(status=200)

    if environ['DEBUG'] != '1':
        app.run('0.0.0.0', 5000) # noqa: S104


if __name__ == '__main__':
    main()
