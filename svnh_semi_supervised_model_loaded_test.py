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


def load_and_predict_with_checkpoints():
    '''
    Loads saved model checkpoints and make prediction on test images
    '''
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


def load_and_predict_with_saved_model():
    '''
    Loads saved as protobuf model and make prediction on a single image
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        # restore save model
        export_dir = './gan-export/1'
        model = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
        # print(model)
        loaded_graph = tf.get_default_graph()

        # get necessary tensors by name
        input_tensor_name = model.signature_def['predict_images'].inputs['images'].name
        input_tensor = loaded_graph.get_tensor_by_name(input_tensor_name)
        output_tensor_name = model.signature_def['predict_images'].outputs['scores'].name
        output_tensor = loaded_graph.get_tensor_by_name(output_tensor_name)

        # make prediction
        image_file_name = './svnh_test_images/image_3.jpg'
        with open(image_file_name, 'rb') as f:
            image = f.read()
            scores = sess.run(output_tensor, {input_tensor: [image]})

        # print results
        print("Scores: {}".format(scores))


def main(_):
    # load_and_predict_with_checkpoints()
    load_and_predict_with_saved_model()


if __name__ == '__main__':
    tf.app.run()
