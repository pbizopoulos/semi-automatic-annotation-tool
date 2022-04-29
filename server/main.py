from flask import Flask, Response, request
from flask_cors import CORS, cross_origin
import io
import tensorflowjs as tfjs
import werkzeug.formparser


class ModelReceiver():
    def stream_factory(self, total_content_length, content_type, filename, content_length=None):
        if filename == 'model.json':
            self.model_json_bytes = io.BytesIO()
            self.model_json_writer = io.BufferedWriter(self.model_json_bytes)
            return self.model_json_writer
        elif filename == 'model.weights.bin':
            self.weight_bytes = io.BytesIO()
            self.weight_writer = io.BufferedWriter(self.weight_bytes)
            return self.weight_writer


def main():
    app = Flask('model-server')
    CORS(app)
    app.config['CORS_HEADER'] = 'Content-Type'
    model_receiver = ModelReceiver()

    @app.route('/upload', methods=['POST'])
    @cross_origin()
    def upload():
        werkzeug.formparser.parse_form_data(request.environ, stream_factory=model_receiver.stream_factory)
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

    app.run('0.0.0.0', 5000)


if __name__ == '__main__':
    main()
