import tensorflow as tf
import numpy as np

import utils


def load_test_images():
    utils.download_train_and_test_data()
    _, testset = utils.load_data_sets()

    idx = np.random.randint(0, testset['X'].shape[3], size=64)
    test_images = testset['X'][:, :, :, idx]
    test_labels = testset['y'][idx]

    test_images = np.rollaxis(test_images, 3)
    test_images = utils.scale(test_images)

    #print(np.squeeze(test_labels))

    return test_images, test_labels


def main(_):
    test_images, test_labels = load_test_images()

    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        saver = tf.train.import_meta_graph('./checkpoints/generator.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./checkpoints'))

        pred_class_tensor = loaded_graph.get_tensor_by_name("pred_class:0")
        inputs_real_tensor = loaded_graph.get_tensor_by_name("input_real:0")
        y_tensor = loaded_graph.get_tensor_by_name("y:0")
        drop_rate_tensor = loaded_graph.get_tensor_by_name("drop_rate:0")
        correct_pred_sum_tensor = loaded_graph.get_tensor_by_name("correct_pred_sum:0")

        correct, pred_class = sess.run(
            [correct_pred_sum_tensor, pred_class_tensor],
            feed_dict={
                inputs_real_tensor: test_images,
                y_tensor: test_labels,
                drop_rate_tensor: 0.})

        print("No. correct predictions: {}".format(correct))
        print("Predicted classes: {}".format(pred_class))


if __name__ == '__main__':
    tf.app.run()
