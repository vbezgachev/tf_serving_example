'''
Send JPEG image to tensorflow_model_server loaded with GAN model.

Hint: the client does not require the complete Tensorflow framework. However
you must create Python files from the protobuf files for:
- tensorflow.core.framework
- tensorflow.core.example
- tensorflow.core.protobuf
- tensorflow_serving.apis
'''

from __future__ import print_function

import time

from argparse import ArgumentParser

# Communication to TensorFlow server via gRPC
from grpc.beta import implementations

# TensorFlow serving stuff to send messages
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2

import six as _six


def parse_args():
    parser = ArgumentParser(description="Request a TensorFlow server for a prediction on the image")
    parser.add_argument("-s", "--server",
                        dest="server",
                        default='172.17.0.2:9000',
                        help="prediction service host:port")
    parser.add_argument("-i", "--image",
                        dest="image",
                        default="",
                        help="path to image in JPEG format",)
    args = parser.parse_args()

    host, port = args.server.split(':')
    
    return host, port, args.image


def main():
    # parse command line arguments
    host, port, image = parse_args()

    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # Send request
    with open(image, 'rb') as f:
        # See prediction_service.proto for gRPC request/response details.
        data = f.read()

        start = time.time()

        request = predict_pb2.PredictRequest()

        # Call GAN model to make prediction on the image
        request.model_spec.name = 'gan'
        request.model_spec.signature_name = 'predict_images'

        # create TensorProto object for a request
        dims = [tensor_shape_pb2.TensorShapeProto.Dim(size=1)]
        tensor_shape_proto = tensor_shape_pb2.TensorShapeProto(dim=dims)
        tensor_proto = tensor_pb2.TensorProto(
            dtype=types_pb2.DT_STRING,
            tensor_shape=tensor_shape_proto,
            string_val=[data])

        # put data into TensorProto and copy them into the request object
        request.inputs['images'].CopyFrom(tensor_proto)

        # call prediction
        result = stub.Predict(request, 60.0)  # 60 secs timeout

        end = time.time()
        time_diff = end - start

        print(result)
        print('time elapased: {}'.format(time_diff))


if __name__ == '__main__':
    main()
