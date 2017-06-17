import os
import shutil

import tensorflow as tf
from tensorflow.python.framework import graph_util

import utils
from gan import GAN



tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/train',
                           """Directory where to read training checkpoints.""")
tf.app.flags.DEFINE_string('output_dir', '/tmp/export',
                           """Directory where to export the model.""")


#def preprocess_image(image_buffer):
#    """Preprocess JPEG encoded bytes to 3D float Tensor."""
#    image = tf.image.decode_jpeg(image_buffer, channels=3)
#    image = utils.scale(image)
#
#    return image


def main(_):

    # Create GAN model
    real_size = (32, 32, 3)
    z_size = 100
    learning_rate = 0.0003
    net = GAN(real_size, z_size, learning_rate, drop_rate=0.)

    # Create saver to restore from checkpoints
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Restore the model from last checkpoints
        saver.restore(sess, tf.train.latest_checkpoint('./checkpoints'))

        # (re-)create export directory
        export_path = 'gan-export'
        if os.path.exists(export_path):
            shutil.rmtree(export_path)

        # create model builder
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)

        # create tensors info
        predict_tensor_inputs_info = tf.saved_model.utils.build_tensor_info(net.input_real)
        predict_tensor_scores_info = tf.saved_model.utils.build_tensor_info(
            net.discriminator_out)

        # build prediction signature
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'images': predict_tensor_inputs_info},
                outputs={'scores': predict_tensor_scores_info},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )
        )

        # save the model
        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_images': prediction_signature
            },
            legacy_init_op=legacy_init_op)

        builder.save()


if __name__ == '__main__':
    tf.app.run()
