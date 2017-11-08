'''
Send JPEG image to tensorflow_model_server loaded with GAN model.

Hint: the code has been compiled together with TensorFlow serving
and not locally. The client is called in the TensorFlow Docker container
'''

from __future__ import print_function

# Communication to TensorFlow server via gRPC
from grpc.beta import implementations
import tensorflow as tf

# TensorFlow serving stuff to send messages
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


# Command line arguments
tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS


def main(_):
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # Send request
    with open(FLAGS.image, 'rb') as f:
        # See prediction_service.proto for gRPC request/response details.
        data = f.read()
        request = predict_pb2.PredictRequest()

        # Call GAN model to make prediction on the image
        request.model_spec.name = 'gan'
        request.model_spec.signature_name = 'predict_images'
        request.inputs['images'].CopyFrom(
            tf.contrib.util.make_tensor_proto(data, shape=[1]))

        result = stub.Predict(request, 60.0)  # 60 secs timeout
        print(result)


if __name__ == '__main__':
    tf.app.run()
