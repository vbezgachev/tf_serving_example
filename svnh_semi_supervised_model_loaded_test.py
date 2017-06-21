import tensorflow as tf
import numpy as np

import utils

'''
Loads the saved GAN model and calls prediction on test images
'''


def load_test_images():
    '''
    Loads 64 random images from SVNH test data sets

    :return: Tuple of (test images, image labels)
    '''
    utils.download_train_and_test_data()
    _, testset = utils.load_data_sets()

    idx = np.random.randint(0, testset['X'].shape[3], size=64)
    test_images = testset['X'][:, :, :, idx]
    test_labels = testset['y'][idx]

    test_images = np.rollaxis(test_images, 3)
    test_images = utils.scale(test_images)

    return test_images, test_labels


def main(_):
    # load test images and labels
    test_images, test_labels = load_test_images()

    # create an empty graph for the session
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # restore save model
        saver = tf.train.import_meta_graph('./checkpoints/generator.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./checkpoints'))

        # get necessary tensors by name
        pred_class_tensor = loaded_graph.get_tensor_by_name("pred_class:0")
        inputs_real_tensor = loaded_graph.get_tensor_by_name("input_real:0")
        y_tensor = loaded_graph.get_tensor_by_name("y:0")
        drop_rate_tensor = loaded_graph.get_tensor_by_name("drop_rate:0")
        correct_pred_sum_tensor = loaded_graph.get_tensor_by_name("correct_pred_sum:0")

        # make prediction
        correct, pred_class = sess.run(
            [correct_pred_sum_tensor, pred_class_tensor],
            feed_dict={
                inputs_real_tensor: test_images,
                y_tensor: test_labels,
                drop_rate_tensor: 0.})

        # print results
        print("No. correct predictions: {}".format(correct))
        print("Predicted classes: {}".format(pred_class))


if __name__ == '__main__':
    tf.app.run()
