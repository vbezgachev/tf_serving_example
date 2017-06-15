import tensorflow as tf

import utils
from gan import GAN


tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/train',
                           """Directory where to read training checkpoints.""")
tf.app.flags.DEFINE_string('output_dir', '/tmp/export',
                           """Directory where to export the model.""")


def preprocess_image(image_buffer):
    """Preprocess JPEG encoded bytes to 3D float Tensor."""
    image = tf.image.decode_jpeg(image_buffer, channels=3)
    image = utils.scale(image)

    return image


def main(_):

    #tf.saved_model.utils.build_tensor_info()

    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        saver = tf.train.import_meta_graph('./checkpoints/generator.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./checkpoints'))

        # Input transformation.
        serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
        feature_configs = {
            'image/encoded': tf.FixedLenFeature(
                shape=[32, 32, 3], dtype=tf.string),
        }
        tf_example = tf.parse_example(serialized_tf_example, feature_configs)
        jpegs = tf_example['image/encoded']
        images = tf.map_fn(preprocess_image, jpegs, dtype=tf.float32)

        # Create GAN model
        real_size = (32, 32, 3)
        z_size = 100
        learning_rate = 0.0003
        net = GAN(real_size, z_size, learning_rate)

        # Restore variables from the last checkpoint
        pred_class_tensor = loaded_graph.get_tensor_by_name("pred_class:0")
        inputs_real_tensor = loaded_graph.get_tensor_by_name("input_real:0")
        drop_rate_tensor = loaded_graph.get_tensor_by_name("drop_rate:0")

        # Build the signature_def_map.
        classification_inputs = tf.saved_model.utils.build_tensor_info(
            serialized_tf_example)
        classification_outputs_classes = tf.saved_model.utils.build_tensor_info(
            net.pred_class)
        classification_outputs_scores = tf.saved_model.utils.build_tensor_info(
            values)


if __name__ == '__main__':
    tf.app.run()
